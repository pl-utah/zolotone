use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use rival::{Discretization, Expr, Hint, Ival, Machine, MachineBuilder};
use rug::{float::Round, Float, Integer, Rational};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

#[derive(Clone)]
struct F64Discretization;

impl Discretization for F64Discretization {
    fn target(&self) -> u32 {
        128
    }

    fn convert(&self, _: usize, value: &Float) -> Float {
        Float::with_val(53, value.to_f64())
    }

    fn distance(&self, _: usize, lo: &Float, hi: &Float) -> usize {
        ordinal_distance_f64(lo.to_f64(), hi.to_f64())
    }
}

#[derive(Clone)]
struct RangeDiscretization;

impl Discretization for RangeDiscretization {
    fn target(&self) -> u32 {
        128
    }

    fn convert(&self, _: usize, value: &Float) -> Float {
        value.clone()
    }

    fn distance(&self, _: usize, _: &Float, _: &Float) -> usize {
        0
    }
}

#[derive(Clone, Copy)]
enum OutputRounding {
    Exact,
    F64,
    Rne,
    Rtz,
    Raz,
}

impl OutputRounding {
    fn convert(self, value: &Float) -> Float {
        match self {
            Self::Exact => value.clone(),
            Self::F64 => Float::with_val(53, value.to_f64()),
            Self::Rne => value.clone().round_even(),
            Self::Rtz => value.clone().trunc(),
            Self::Raz if value.is_sign_negative() => value.clone().floor(),
            Self::Raz => value.clone().ceil(),
        }
    }
}

/// Resolve output 0 as a Boolean and output 1 using its selected discretization.
#[derive(Clone)]
struct EvaluationDiscretization {
    rounding: OutputRounding,
    range_mode: Arc<AtomicBool>,
}

impl Discretization for EvaluationDiscretization {
    fn target(&self) -> u32 {
        128
    }

    fn convert(&self, idx: usize, value: &Float) -> Float {
        if idx == 0 || (matches!(self.rounding, OutputRounding::F64)
            && self.range_mode.load(Ordering::Relaxed)) {
            return value.clone();
        }
        // Integer modes preserve endpoint precision so distinct codes cannot
        // merge; F64 mode rounds only for point-convergence checks.
        self.rounding.convert(value)
    }

    fn distance(&self, idx: usize, lo: &Float, hi: &Float) -> usize {
        if self.range_mode.load(Ordering::Relaxed) {
            return 0;
        }
        if idx == 0 {
            // A Boolean must resolve exactly; never treat false/true as an
            // adjacent rounding boundary.
            return if lo == hi { 0 } else { 2 };
        }
        if matches!(self.rounding, OutputRounding::F64) {
            return ordinal_distance_f64(lo.to_f64(), hi.to_f64());
        }
        if !lo.is_finite() || !hi.is_finite() {
            return 2;
        }
        if lo == hi {
            return 0;
        }
        let difference = Float::with_val(lo.prec().max(hi.prec()), hi - lo);
        if difference == 1 || difference == -1 {
            1
        } else {
            2
        }
    }
}

fn ordinal_distance_f64(x: f64, y: f64) -> usize {
    if x == y {
        return 0;
    }
    let to_ordinal = |value: f64| -> i64 {
        if value == 0.0 {
            return 0;
        }
        let bits = value.to_bits() as i64;
        if bits < 0 {
            !bits
        } else {
            bits
        }
    };
    to_ordinal(y).wrapping_sub(to_ordinal(x)).unsigned_abs() as usize
}

#[pyclass(module = "zolotone._rival3")]
struct RivalHints {
    hints: Vec<Hint>,
}

#[pymethods]
impl RivalHints {
    fn __len__(&self) -> usize {
        self.hints.len()
    }
}

#[pyclass(module = "zolotone._rival3")]
struct RawRivalMachine {
    machine: Machine<F64Discretization>,
    range_machine: Machine<RangeDiscretization>,
}

#[pyclass(module = "zolotone._rival3")]
struct RawEvaluator {
    machine: Machine<EvaluationDiscretization>,
    rounding: OutputRounding,
    range_mode: Arc<AtomicBool>,
}

#[pyclass(module = "zolotone._rival3")]
struct RawFloatEvaluator {
    evaluator: RawEvaluator,
}

type FloatEvaluationRange = ((bool, bool), (f64, f64), bool, bool, bool, bool);

#[pymethods]
impl RawFloatEvaluator {
    fn apply_point(&mut self, point: Vec<String>) -> PyResult<(bool, f64, bool, bool, bool, bool)> {
        let args = self.evaluator.code_args(
            point.into_iter().map(|code| (code.clone(), code)).collect()
        )?;
        let outputs = self.evaluator.machine.apply(&args, None, 10_000)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let precondition = &outputs[0];
        let output = &outputs[1];
        Ok((
            !precondition.lo().is_zero(), output.lo().to_f64(),
            precondition.error_flags().partial(), output.error_flags().partial(),
            precondition.error_flags().total(), output.error_flags().total(),
        ))
    }

    fn apply_range(&mut self, intervals: Vec<(String, String)>) -> PyResult<Option<FloatEvaluationRange>> {
        let args = self.evaluator.code_args(intervals)?;
        self.evaluator.range_mode.store(true, Ordering::Relaxed);
        let result = self.evaluator.machine.apply(&args, None, 1);
        self.evaluator.range_mode.store(false, Ordering::Relaxed);
        let outputs = match result {
            Ok(outputs) => outputs,
            Err(_) => return Ok(None),
        };
        let precondition = &outputs[0];
        let output = &outputs[1];
        Ok(Some((
            (!precondition.lo().is_zero(), !precondition.hi().is_zero()),
            (output.lo().to_f64_round(Round::Down), output.hi().to_f64_round(Round::Up)),
            precondition.error_flags().partial(), output.error_flags().partial(),
            precondition.error_flags().total(), output.error_flags().total(),
        )))
    }
}

type EvaluationRange = ((bool, bool), Option<(String, String)>, bool, bool, bool, bool);

impl RawEvaluator {
    fn code_args(&self, intervals: Vec<(String, String)>) -> PyResult<Vec<Ival>> {
        if intervals.len() != self.machine.argument_count() {
            return Err(PyValueError::new_err(format!(
                "expected {} inputs, got {}",
                self.machine.argument_count(),
                intervals.len(),
            )));
        }
        intervals
            .into_iter()
            .map(|(lower, upper)| {
                let lower = parse_integer(&lower, "lower input code")?;
                let upper = parse_integer(&upper, "upper input code")?;
                if lower > upper {
                    return Err(PyValueError::new_err("reversed input code interval"));
                }
                let precision = self
                    .machine
                    .argument_precision()
                    .max(lower.significant_bits())
                    .max(upper.significant_bits());
                Ok(Ival::from_lo_hi(
                    Float::with_val(precision, lower),
                    Float::with_val(precision, upper),
                ))
            })
            .collect()
    }
}

fn rect_args(rect: Vec<(f64, f64)>, precision: u32) -> Vec<Ival> {
    rect.into_iter()
        .map(|(lo, hi)| {
            Ival::from_lo_hi(
                Float::with_val(precision, lo),
                Float::with_val(precision, hi),
            )
        })
        .collect()
}

#[pymethods]
impl RawRivalMachine {
    fn domain_errors(&mut self, rect: Vec<(f64, f64)>) -> PyResult<Vec<(bool, bool)>> {
        let rival_rect = rect_args(rect, self.range_machine.argument_precision());

        let outputs = self
            .range_machine
            .apply(&rival_rect, None, 1)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(outputs
            .iter()
            .map(|output| (!output.lo().is_zero(), !output.hi().is_zero()))
            .collect())
    }

    fn apply_with_hints(
        &mut self,
        rect: Vec<(f64, f64)>,
        hints: Option<PyRef<'_, RivalHints>>,
    ) -> PyResult<((bool, bool), RivalHints)> {
        let rival_rect = rect_args(rect, self.machine.argument_precision());

        let hint_slice = hints.as_ref().map(|h| h.hints.as_slice());
        let (status, next_hints, _) = self.machine.analyze_with_hints(&rival_rect, hint_slice);
        let status_interval = (!status.lo().is_zero(), !status.hi().is_zero());
        Ok((status_interval, RivalHints { hints: next_hints }))
    }

    fn apply_range(&mut self, rect: Vec<(f64, f64)>) -> PyResult<(f64, f64)> {
        let rival_rect = rect_args(rect, self.machine.argument_precision());

        let outputs = match self.range_machine.apply(&rival_rect, None, 1) {
            Ok(outputs) => outputs,
            // An unsamplable rectangle has no representable output range. Let
            // the Python range-analysis layer convert this sentinel to None.
            Err(_) => return Ok((f64::NAN, f64::NAN)),
        };
        let range = outputs
            .first()
            .ok_or_else(|| PyValueError::new_err("range machine returned no outputs"))?;
        let output_range = (
            range.lo().to_f64_round(Round::Down),
            range.hi().to_f64_round(Round::Up),
        );
        Ok(output_range)
    }
}

#[pymethods]
impl RawEvaluator {
    fn apply_range(
        &mut self,
        intervals: Vec<(String, String)>,
    ) -> PyResult<Option<EvaluationRange>> {
        let args = self.code_args(intervals)?;
        // The same compiled evaluator produces a single-pass enclosure here;
        // point evaluation still refines until its output code is resolved.
        self.range_mode.store(true, Ordering::Relaxed);
        let result = self.machine.apply(&args, None, 1);
        self.range_mode.store(false, Ordering::Relaxed);
        let outputs = match result {
            Ok(outputs) => outputs,
            Err(_) => return Ok(None),
        };
        let precondition = &outputs[0];
        let output = &outputs[1];
        let lower = self.rounding.convert(output.lo()).to_rational();
        let upper = self.rounding.convert(output.hi()).to_rational();
        Ok(Some((
            (!precondition.lo().is_zero(), !precondition.hi().is_zero()),
            lower
                .zip(upper)
                .map(|(lo, hi)| (lo.to_string(), hi.to_string())),
            precondition.error_flags().partial(),
            output.error_flags().partial(),
            precondition.error_flags().total(),
            output.error_flags().total(),
        )))
    }

    /// Evaluate both outputs using exact integer input codes.
    /// Strings avoid host integer-width limits.
    fn apply_point(&mut self, point: Vec<String>) -> PyResult<(bool, Option<String>, bool, bool, bool, bool)> {
        let args = self.code_args(point.into_iter().map(|code| (code.clone(), code)).collect())?;
        let outputs = self
            .machine
            .apply(&args, None, 10_000)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;

        let feasible = &outputs[0];
        let output = &outputs[1];
        if matches!(self.rounding, OutputRounding::Exact)
            && !output.error_flags().partial()
            && output.lo().is_finite()
            && !output.lo().is_integer()
        {
            return Err(PyValueError::new_err(
                "Exact integer evaluation produced a non-integer output",
            ));
        }
        // Preserve flags and non-finite values for the caller to interpret.
        // Error-free outputs already agree after discretization.
        let rounded = self.rounding.convert(output.lo());
        let integer = rounded.to_integer().map(|value| value.to_string());
        Ok((
            !feasible.lo().is_zero(),
            integer,
            feasible.error_flags().partial(),
            output.error_flags().partial(),
            feasible.error_flags().total(),
            output.error_flags().total(),
        ))
    }
}

#[pyfunction]
fn build_machine(exprs: &PyList, free_vars: Vec<String>) -> PyResult<RawRivalMachine> {
    if exprs.is_empty() {
        return Err(PyValueError::new_err("exprs must be non-empty"));
    }

    let mut rival_exprs = Vec::with_capacity(exprs.len());
    for expr in exprs.iter() {
        rival_exprs.push(parse_expr(expr)?);
    }

    let range_machine =
        MachineBuilder::new(RangeDiscretization).build(rival_exprs.clone(), free_vars.clone());
    let machine = MachineBuilder::new(F64Discretization).build(rival_exprs, free_vars);

    Ok(RawRivalMachine {
        machine,
        range_machine,
    })
}

#[pyfunction]
#[pyo3(signature = (expr, rounding_mode=None, precondition=None, free_vars=None))]
fn build_integer_evaluator(
    expr: &PyAny,
    rounding_mode: Option<&str>,
    precondition: Option<&PyAny>,
    free_vars: Option<Vec<String>>,
) -> PyResult<RawEvaluator> {
    let rounding = match rounding_mode {
        None => OutputRounding::Exact,
        Some("RNE") => OutputRounding::Rne,
        Some("RTZ") => OutputRounding::Rtz,
        Some("RAZ") => OutputRounding::Raz,
        Some(mode) => {
            return Err(PyValueError::new_err(format!(
                "unsupported output rounding mode {mode:?}"
            )))
        }
    };
    build_evaluator(expr, rounding, precondition, free_vars)
}

#[pyfunction]
#[pyo3(signature = (expr, precondition=None, free_vars=None))]
fn build_float_evaluator(
    expr: &PyAny,
    precondition: Option<&PyAny>,
    free_vars: Option<Vec<String>>,
) -> PyResult<RawFloatEvaluator> {
    Ok(RawFloatEvaluator {
        evaluator: build_evaluator(expr, OutputRounding::F64, precondition, free_vars)?,
    })
}

fn build_evaluator(
    expr: &PyAny,
    rounding: OutputRounding,
    precondition: Option<&PyAny>,
    free_vars: Option<Vec<String>>,
) -> PyResult<RawEvaluator> {
    let rival_expr = parse_expr(expr)?;
    let range_mode = Arc::new(AtomicBool::new(false));
    let disc = EvaluationDiscretization {
        rounding,
        range_mode: range_mode.clone(),
    };
    let precondition = match precondition {
        Some(precondition) => parse_expr(precondition)?,
        None => Expr::Literal(Float::with_val(2, 1)),
    };
    // Rival otherwise rejects the entire call when every output has a total
    // domain error. A valid sentinel lets us return both outputs' error flags.
    let expressions = vec![
        precondition,
        rival_expr,
        Expr::Literal(Float::with_val(2, 0)),
    ];
    Ok(RawEvaluator {
        rounding,
        range_mode,
        machine: MachineBuilder::new(disc).build(
            expressions,
            free_vars.unwrap_or_else(|| vec!["x".to_string()]),
        ),
    })
}

fn dict_get<'py>(dict: &'py PyDict, key: &str) -> PyResult<&'py PyAny> {
    dict.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("Rival IR node missing key {key:?}")))
}

fn parse_expr(obj: &PyAny) -> PyResult<Expr> {
    let dict = obj
        .downcast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("Rival IR node must be a dict"))?;
    let op: &str = dict_get(dict, "op")?.extract()?;

    match op {
        "var" => Ok(Expr::Var(dict_get(dict, "name")?.extract()?)),
        "real_lit" => parse_real_lit(dict),
        "pi" => Ok(Expr::Pi),
        "bool_lit" => {
            let value: bool = dict_get(dict, "value")?.extract()?;
            Ok(Expr::Literal(Float::with_val(2, if value { 1 } else { 0 })))
        }
        "add" => parse_binary(dict, Expr::Add),
        "sub" => parse_binary(dict, Expr::Sub),
        "mul" => parse_binary(dict, Expr::Mul),
        "neg" => parse_unary(dict, Expr::Neg),
        "abs" => parse_unary(dict, Expr::Fabs),
        "pow2" => parse_unary(dict, Expr::Pow2),
        "sqrt" => parse_unary(dict, Expr::Sqrt),
        "cbrt" => parse_unary(dict, Expr::Cbrt),
        "exp" => parse_unary(dict, Expr::Exp),
        "exp2" => parse_unary(dict, Expr::Exp2),
        "expm1" => parse_unary(dict, Expr::Expm1),
        "log" => parse_unary(dict, Expr::Log),
        "log2" => parse_unary(dict, Expr::Log2),
        "log10" => parse_unary(dict, Expr::Log10),
        "log1p" => parse_unary(dict, Expr::Log1p),
        "logb" => parse_unary(dict, Expr::Logb),
        "sin" => parse_unary(dict, Expr::Sin),
        "cos" => parse_unary(dict, Expr::Cos),
        "tan" => parse_unary(dict, Expr::Tan),
        "asin" => parse_unary(dict, Expr::Asin),
        "acos" => parse_unary(dict, Expr::Acos),
        "atan" => parse_unary(dict, Expr::Atan),
        "sinh" => parse_unary(dict, Expr::Sinh),
        "cosh" => parse_unary(dict, Expr::Cosh),
        "tanh" => parse_unary(dict, Expr::Tanh),
        "asinh" => parse_unary(dict, Expr::Asinh),
        "acosh" => parse_unary(dict, Expr::Acosh),
        "atanh" => parse_unary(dict, Expr::Atanh),
        "erf" => parse_unary(dict, Expr::Erf),
        "erfc" => parse_unary(dict, Expr::Erfc),
        "round_even" => parse_unary(dict, Expr::Round),
        "floor" => parse_unary(dict, Expr::Floor),
        "ceil" => parse_unary(dict, Expr::Ceil),
        "trunc" => parse_unary(dict, Expr::Trunc),
        "pow" => parse_binary(dict, Expr::Pow),
        "min" => parse_binary(dict, Expr::Fmin),
        "max" => parse_binary(dict, Expr::Fmax),
        "if" => parse_ternary(dict, Expr::If),
        "eq" => parse_binary(dict, Expr::Eq),
        "ne" => parse_binary(dict, Expr::Ne),
        "lt" => parse_binary(dict, Expr::Lt),
        "le" => parse_binary(dict, Expr::Le),
        "gt" => parse_binary(dict, Expr::Gt),
        "ge" => parse_binary(dict, Expr::Ge),
        "bool_eq" => parse_binary(dict, Expr::Eq),
        "not" => parse_unary(dict, Expr::Not),
        "error" => parse_unary(dict, Expr::Error),
        "or" => parse_binary(dict, Expr::Or),
        "and" => parse_binary(dict, Expr::And),
        "assert" => parse_unary(dict, Expr::Assert),
        _ => Err(PyValueError::new_err(format!(
            "unsupported Rival IR op {op:?}"
        ))),
    }
}

fn parse_real_lit(dict: &PyDict) -> PyResult<Expr> {
    let numerator_text: String = dict_get(dict, "num")?.extract()?;
    let denominator_text: String = dict_get(dict, "den")?.extract()?;

    let numerator = parse_integer(&numerator_text, "num")?;
    let denominator = parse_integer(&denominator_text, "den")?;
    if denominator.cmp0() == std::cmp::Ordering::Equal {
        return Err(PyValueError::new_err(
            "real_lit denominator must be non-zero",
        ));
    }

    Ok(Expr::Rational(Rational::from((numerator, denominator))))
}

fn parse_integer(value: &str, field: &str) -> PyResult<Integer> {
    Integer::parse(value)
        .map(Integer::from)
        .map_err(|_| PyValueError::new_err(format!("invalid integer in {field}: {value:?}")))
}

fn parse_unary(dict: &PyDict, constructor: fn(Box<Expr>) -> Expr) -> PyResult<Expr> {
    Ok(constructor(Box::new(parse_expr(dict_get(dict, "arg")?)?)))
}

fn parse_binary(dict: &PyDict, constructor: fn(Box<Expr>, Box<Expr>) -> Expr) -> PyResult<Expr> {
    Ok(constructor(
        Box::new(parse_expr(dict_get(dict, "lhs")?)?),
        Box::new(parse_expr(dict_get(dict, "rhs")?)?),
    ))
}

fn parse_ternary(
    dict: &PyDict,
    constructor: fn(Box<Expr>, Box<Expr>, Box<Expr>) -> Expr,
) -> PyResult<Expr> {
    Ok(constructor(
        Box::new(parse_expr(dict_get(dict, "cond")?)?),
        Box::new(parse_expr(dict_get(dict, "on_true")?)?),
        Box::new(parse_expr(dict_get(dict, "on_false")?)?),
    ))
}

#[pymodule]
fn _rival3(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<RawRivalMachine>()?;
    module.add_class::<RawEvaluator>()?;
    module.add_class::<RawFloatEvaluator>()?;
    module.add_class::<RivalHints>()?;
    module.add_function(wrap_pyfunction!(build_machine, module)?)?;
    module.add_function(wrap_pyfunction!(build_integer_evaluator, module)?)?;
    module.add_function(wrap_pyfunction!(build_float_evaluator, module)?)?;
    module.add("__doc__", "Native Rival3 bridge for zolotone")?;
    Ok(())
}
