#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <mpfi.h>
#include <stdlib.h>

typedef struct { mpfi_t x; } Interval;
static const char *tag = "zolotone.mpfi.interval";
static void destroy(PyObject *obj) {
    Interval *v = PyCapsule_GetPointer(obj, tag);
    if (v) { mpfi_clear(v->x); free(v); }
}
static Interval *get(PyObject *obj) { return PyCapsule_GetPointer(obj, tag); }
static PyObject *allocate(long precision) {
    if (precision < MPFR_PREC_MIN || precision > MPFR_PREC_MAX) {
        PyErr_SetString(PyExc_ValueError, "invalid MPFI precision"); return NULL;
    }
    Interval *v = malloc(sizeof(*v));
    if (!v) return PyErr_NoMemory();
    mpfi_init2(v->x, precision);
    PyObject *obj = PyCapsule_New(v, tag, destroy);
    if (!obj) { mpfi_clear(v->x); free(v); }
    return obj;
}
static PyObject *create(PyObject *self, PyObject *args) {
    long precision; PyObject *value;
    if (!PyArg_ParseTuple(args, "lO", &precision, &value)) return NULL;
    PyObject *obj = allocate(precision);
    if (!obj) return NULL;
    Interval *v = get(obj);
    if (PyFloat_Check(value)) {
        mpfi_set_d(v->x, PyFloat_AS_DOUBLE(value));
    } else if (PyUnicode_Check(value)) {
        const char *s = PyUnicode_AsUTF8(value);
        if (!s) { Py_DECREF(obj); return NULL; }
        mpq_t q; mpq_init(q);
        int bad = mpq_set_str(q, s, 16);
        if (bad || mpz_sgn(mpq_denref(q)) == 0) {
            mpq_clear(q); Py_DECREF(obj);
            PyErr_SetString(PyExc_ValueError, "invalid rational"); return NULL;
        }
        mpq_canonicalize(q); mpfi_set_q(v->x, q); mpq_clear(q);
    } else {
        Py_DECREF(obj); PyErr_SetString(PyExc_TypeError, "expected float or hexadecimal rational"); return NULL;
    }
    return obj;
}
static PyObject *unary(PyObject *self, PyObject *args) {
    const char *op; PyObject *a;
    if (!PyArg_ParseTuple(args, "sO", &op, &a)) return NULL;
    Interval *x = get(a); if (!x) return NULL;
    PyObject *obj = allocate(mpfi_get_prec(x->x)); if (!obj) return NULL;
    Interval *r = get(obj);
    if (0) {}
    else if (!strcmp(op, "neg")) mpfi_neg(r->x, x->x);
    else if (!strcmp(op, "abs")) mpfi_abs(r->x, x->x);
    else if (!strcmp(op, "sqr")) mpfi_sqr(r->x, x->x);
    else if (!strcmp(op, "sqrt")) mpfi_sqrt(r->x, x->x);
    else if (!strcmp(op, "exp")) mpfi_exp(r->x, x->x);
    else if (!strcmp(op, "exp2")) mpfi_exp2(r->x, x->x);
    else if (!strcmp(op, "log")) mpfi_log(r->x, x->x);
    else if (!strcmp(op, "log2")) mpfi_log2(r->x, x->x);
    else if (!strcmp(op, "log10")) mpfi_log10(r->x, x->x);
    else if (!strcmp(op, "sin")) mpfi_sin(r->x, x->x);
    else if (!strcmp(op, "cos")) mpfi_cos(r->x, x->x);
    else if (!strcmp(op, "tan")) mpfi_tan(r->x, x->x);
    else if (!strcmp(op, "asin")) mpfi_asin(r->x, x->x);
    else if (!strcmp(op, "acos")) mpfi_acos(r->x, x->x);
    else if (!strcmp(op, "atan")) mpfi_atan(r->x, x->x);
    else if (!strcmp(op, "sinh")) mpfi_sinh(r->x, x->x);
    else if (!strcmp(op, "cosh")) mpfi_cosh(r->x, x->x);
    else if (!strcmp(op, "tanh")) mpfi_tanh(r->x, x->x);
    else if (!strcmp(op, "asinh")) mpfi_asinh(r->x, x->x);
    else if (!strcmp(op, "acosh")) mpfi_acosh(r->x, x->x);
    else if (!strcmp(op, "atanh")) mpfi_atanh(r->x, x->x);
    else if (!strcmp(op, "pi")) mpfi_const_pi(r->x);
    else { Py_DECREF(obj); PyErr_SetString(PyExc_ValueError, "unknown unary operation"); return NULL; }
    return obj;
}
static PyObject *binary(PyObject *self, PyObject *args) {
    const char *op; PyObject *a, *b;
    if (!PyArg_ParseTuple(args, "sOO", &op, &a, &b)) return NULL;
    Interval *x = get(a); if (!x) return NULL;
    Interval *y = get(b); if (!y) return NULL;
    mpfr_prec_t p = mpfi_get_prec(x->x);
    if (p != mpfi_get_prec(y->x)) { PyErr_SetString(PyExc_ValueError, "precision mismatch"); return NULL; }
    PyObject *obj = allocate(p); if (!obj) return NULL;
    Interval *r = get(obj);
    if (0) {}
    else if (!strcmp(op, "add")) mpfi_add(r->x, x->x, y->x);
    else if (!strcmp(op, "sub")) mpfi_sub(r->x, x->x, y->x);
    else if (!strcmp(op, "mul")) mpfi_mul(r->x, x->x, y->x);
    else if (!strcmp(op, "div")) mpfi_div(r->x, x->x, y->x);
    else if (!strcmp(op, "pow")) {
        mpfi_log(r->x, x->x); mpfi_mul(r->x, r->x, y->x); mpfi_exp(r->x, r->x);
    } else if (!strcmp(op, "min") || !strcmp(op, "max")) {
        mpfr_t lo, hi, yl, yr;
        mpfr_inits2(p, lo, hi, yl, yr, (mpfr_ptr)0);
        mpfi_get_left(lo, x->x); mpfi_get_right(hi, x->x);
        mpfi_get_left(yl, y->x); mpfi_get_right(yr, y->x);
        if (!strcmp(op, "min")) {
            mpfr_min(lo, lo, yl, MPFR_RNDD); mpfr_min(hi, hi, yr, MPFR_RNDU);
        } else {
            mpfr_max(lo, lo, yl, MPFR_RNDD); mpfr_max(hi, hi, yr, MPFR_RNDU);
        }
        mpfi_interv_fr(r->x, lo, hi);
        mpfr_clears(lo, hi, yl, yr, (mpfr_ptr)0);
    } else { Py_DECREF(obj); PyErr_SetString(PyExc_ValueError, "unknown binary operation"); return NULL; }
    return obj;
}
static PyObject *endpoint(PyObject *self, PyObject *args) {
    PyObject *a; int upper;
    if (!PyArg_ParseTuple(args, "Op", &a, &upper)) return NULL;
    Interval *x = get(a); if (!x) return NULL;
    mpfr_t f; mpfr_init2(f, mpfi_get_prec(x->x));
    if (upper) mpfi_get_right(f, x->x); else mpfi_get_left(f, x->x);
    PyObject *result;
    if (!mpfr_number_p(f)) result = PyFloat_FromDouble(mpfr_get_d(f, MPFR_RNDN));
    else {
        mpfr_exp_t exponent;
        char *digits = mpfr_get_str(NULL, &exponent, 2, 0, f, MPFR_RNDN);
        if (!digits) { mpfr_clear(f); return PyErr_NoMemory(); }
        result = Py_BuildValue("(sl)", digits, (long)exponent);
        mpfr_free_str(digits);
    }
    mpfr_clear(f); return result;
}
static PyMethodDef methods[] = {
    {"create", create, METH_VARARGS, "Create an owned MPFI interval."},
    {"unary", unary, METH_VARARGS, "Evaluate an MPFI unary operation."},
    {"binary", binary, METH_VARARGS, "Evaluate an interval binary operation."},
    {"endpoint", endpoint, METH_VARARGS, "Return an exact binary endpoint."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_zolotone_mpfi", NULL, -1, methods};
PyMODINIT_FUNC PyInit__zolotone_mpfi(void) { return PyModule_Create(&module); }
