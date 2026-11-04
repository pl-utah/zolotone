//! Arbitrary ordered outputs, with no special precondition position.
use super::*;

#[derive(Clone)]
struct OutputDiscretizations {
    modes: Vec<OutputRounding>,
    range_mode: Arc<AtomicBool>,
    target_precision: u32,
}

impl Discretization for OutputDiscretizations {
    fn target(&self) -> u32 {
        self.target_precision
    }

    fn convert(&self, idx: usize, value: &Float) -> Float {
        self.modes[idx].convert(value)
    }

    fn distance(&self, idx: usize, lo: &Float, hi: &Float) -> usize {
        // This is a little patch - ideally Rival should have range evaluations
        if self.range_mode.load(Ordering::Relaxed) || lo == hi {
            return 0;
        }
        match self.modes[idx] {
            OutputRounding::F64 => ordinal_distance_f64(lo.to_f64(), hi.to_f64()),
            OutputRounding::Rne => integer_distance(lo, hi),
            _ => 2,
        }
    }
}

// lower, upper, partial domain error, total domain error, resolved
type Output = (Option<String>, Option<String>, bool, bool, bool);

const PROFILE_CAPACITY: usize = 1000;

// Rival passes endpoints after conversion: equal integer codes converge,
// adjacent codes straddle a rounding boundary, and wider gaps need refinement.
fn integer_distance(lo: &Float, hi: &Float) -> usize {
    Float::with_val(lo.prec().max(hi.prec()).max(usize::BITS), hi - lo)
        .abs()
        .to_integer()
        .and_then(|difference| difference.to_usize())
        .unwrap_or(usize::MAX)
}

#[pyclass(module = "zolotone._rival3")]
pub struct RawCompiledMachine {
    machine: Machine<OutputDiscretizations>,
    modes: Vec<OutputRounding>,
    range_mode: Arc<AtomicBool>,
}

// Builds machine
fn build_machine(
    expressions: Vec<Expr>,
    modes: Vec<OutputRounding>,
    free_vars: Vec<String>,
    range_mode: Arc<AtomicBool>,
    max_precision: u32,
) -> Machine<OutputDiscretizations> {
    MachineBuilder::new(OutputDiscretizations {
        modes,
        range_mode,
        target_precision: 128,
    })
    .max_precision(max_precision)
    .enable_profiling(true)
    .profile_capacity(PROFILE_CAPACITY)
    .build(expressions, free_vars)
}

// That's ugly
fn output_value(value: &Float, mode: OutputRounding) -> PyResult<String> {
    let rounded = mode.convert(value);
    match mode {
        OutputRounding::F64 => Ok(rounded.to_f64().to_string()),
        _ => rounded
            .to_integer()
            .map(|value| value.to_string())
            .ok_or_else(|| {
                PyValueError::new_err("Non-finite output cannot be converted to an integer")
            }),
    }
}

// Returns a tuple of (lo, hi, error_possible, error_total, resolved?)
fn output_result(output: &Ival, mode: OutputRounding) -> PyResult<Output> {
    let partial = output.error_flags().partial();
    let total = output.error_flags().total();
    if partial {
        return Ok((None, None, partial, total, false));
    }
    let lo = output_value(output.lo(), mode)?;
    let hi = output_value(output.hi(), mode)?;
    // Successful point evaluation already guarantees discretized agreement;
    // range evaluation intentionally returns an enclosure instead.
    Ok((Some(lo), Some(hi), partial, total, true))
}

impl RawCompiledMachine {
    fn evaluate(
        &mut self,
        intervals: Vec<(String, String)>,
        range: bool,
        max_iterations: usize,
    ) -> PyResult<Vec<Output>> {
        if max_iterations == 0 {
            return Err(PyValueError::new_err("max_iterations must be positive"));
        }
        if intervals.len() != self.machine.argument_count() {
            return Err(PyValueError::new_err(format!(
                "expected {} inputs, got {}",
                self.machine.argument_count(),
                intervals.len()
            )));
        }
        let args: Vec<Ival> = intervals
            .into_iter()
            .map(|(lo, hi)| {
                let parse = |text: &str| -> PyResult<Rational> {
                    let value = text
                        .parse::<Rational>()
                        .map_err(|_| PyValueError::new_err("invalid input value"))?;
                    // Python encodes integers and finite binary floats exactly.
                    if value.denom().count_ones() != Some(1) {
                        return Err(PyValueError::new_err("inputs must be dyadic rationals"));
                    }
                    Ok(value)
                };
                let lo = parse(&lo)?;
                let hi = parse(&hi)?;
                if lo > hi {
                    return Err(PyValueError::new_err("reversed input interval"));
                }
                // Dyadic inputs are exact when both numerators fit in the
                // significand, independently of the computation's budget.
                let precision = 128
                    .max(lo.numer().significant_bits())
                    .max(hi.numer().significant_bits());
                Ok(Ival::from_lo_hi(
                    Float::with_val(precision, lo),
                    Float::with_val(precision, hi),
                ))
            })
            .collect::<PyResult<_>>()?;
        self.range_mode.store(range, Ordering::Relaxed);
        let iterations = if range { 1 } else { max_iterations };
        let result = self.machine.apply(&args, None, iterations);
        self.range_mode.store(false, Ordering::Relaxed);
        result
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .iter()
            .zip(&self.modes)
            .map(|(output, mode)| output_result(output, *mode))
            .collect()
    }
}

#[pymethods]
impl RawCompiledMachine {
    fn highest_precision(&self) -> PyResult<u32> {
        let records = self.machine.execution_records();
        if records.len() == PROFILE_CAPACITY {
            return Err(PyValueError::new_err(
                "Rival profiling buffer is full; highest precision may be missing",
            ));
        }
        Ok(records
            .iter()
            .filter(|record| record.number >= 0)
            .map(|record| record.precision)
            .max()
            // A machine returning only an argument executes no instructions.
            .unwrap_or_else(|| self.machine.argument_precision()))
    }

    #[pyo3(signature = (point, max_iterations=10_000))]
    fn apply_point(&mut self, point: Vec<String>, max_iterations: usize) -> PyResult<Vec<Output>> {
        self.evaluate(
            point.into_iter().map(|x| (x.clone(), x)).collect(),
            false,
            max_iterations,
        )
    }

    fn apply_range(&mut self, intervals: Vec<(String, String)>) -> PyResult<Vec<Output>> {
        self.evaluate(intervals, true, 1)
    }
}

#[pyfunction]
#[pyo3(signature = (expressions, discretizations, free_vars, max_precision=10_000))]
pub fn rival_compile(
    expressions: &PyList,
    discretizations: Vec<String>,
    free_vars: Vec<String>,
    max_precision: u32,
) -> PyResult<RawCompiledMachine> {
    // Some initial checks
    if max_precision < 20 {
        return Err(PyValueError::new_err(
            "max_precision must be at least 20 bits",
        ));
    }
    if expressions.is_empty() || expressions.len() != discretizations.len() {
        return Err(PyValueError::new_err(
            "provide one discretization per expression, with at least one expression",
        ));
    }
    // Discs (defined in lib.rs)
    let modes = discretizations
        .iter()
        .map(|disc| match disc.as_str() {
            "boolean" => Ok(OutputRounding::Exact),
            "integer" => Ok(OutputRounding::Rne),
            "float" => Ok(OutputRounding::F64),
            _ => Err(PyValueError::new_err(format!(
                "unknown discretization: {disc}"
            ))),
        })
        .collect::<PyResult<Vec<_>>>()?;
    let expressions = expressions
        .iter()
        .map(parse_expr)
        .collect::<PyResult<Vec<_>>>()?;
    let range_mode = Arc::new(AtomicBool::new(false));
    let machine = build_machine(
        expressions,
        modes.clone(),
        free_vars,
        range_mode.clone(),
        max_precision,
    );
    Ok(RawCompiledMachine {
        machine,
        modes,
        range_mode,
    })
}
