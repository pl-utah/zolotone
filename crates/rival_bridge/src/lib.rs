use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use rival::{Discretization, Expr, Hint, Ival, Machine, MachineBuilder};
use rug::{Float, Integer, Rational};

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
}

#[pymethods]
impl RawRivalMachine {
    fn apply_with_hints(
        &mut self,
        rect: Vec<(f64, f64)>,
        hints: Option<PyRef<'_, RivalHints>>,
    ) -> PyResult<((bool, bool), RivalHints)> {
        let precision = self.machine.argument_precision();
        let rival_rect = rect
            .into_iter()
            .map(|(lo, hi)| {
                Ival::from_lo_hi(
                    Float::with_val(precision, lo),
                    Float::with_val(precision, hi),
                )
            })
            .collect::<Vec<_>>();

        let hint_slice = hints.as_ref().map(|h| h.hints.as_slice());
        let (status, next_hints, _) = self.machine.analyze_with_hints(&rival_rect, hint_slice);
        let status_interval = (!status.lo().is_zero(), !status.hi().is_zero());
        Ok((status_interval, RivalHints { hints: next_hints }))
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

    let machine = MachineBuilder::new(F64Discretization).build(rival_exprs, free_vars);

    Ok(RawRivalMachine { machine })
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
        "bool_lit" => {
            let value: bool = dict_get(dict, "value")?.extract()?;
            Ok(Expr::Literal(Float::with_val(2, if value { 1 } else { 0 })))
        }
        "add" => parse_binary(dict, Expr::Add),
        "sub" => parse_binary(dict, Expr::Sub),
        "mul" => parse_binary(dict, Expr::Mul),
        "neg" => parse_unary(dict, Expr::Neg),
        "abs" => parse_unary(dict, Expr::Fabs),
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
    module.add_class::<RivalHints>()?;
    module.add_function(wrap_pyfunction!(build_machine, module)?)?;
    module.add("__doc__", "Native Rival3 bridge for zolotone")?;
    Ok(())
}
