from zolotone import *


def Test(expect=None):
    def wrapper(spec):
        try:
            generated = Autogenerate(name=spec.__name__, spec=spec)
        except Exception as error:
            if expect is None:
                raise AssertionError(
                    f"{spec.__name__}: expected success, got "
                    f"{type(error).__name__}: {error}"
                ) from error
            if not isinstance(error, expect):
                raise AssertionError(
                    f"{spec.__name__}: expected {expect.__name__}, got "
                    f"{type(error).__name__}: {error}"
                ) from error
            print(f"{spec.__name__}: passed (raised {str(error)})")
            return spec

        if expect is not None:
            raise AssertionError(
                f"{spec.__name__}: expected {expect.__name__}, but succeeded"
            )

        generated.print_tree(depth=1)
        print(f"{spec.__name__}: passed")
        return spec

    return wrapper


@Test()
def spec1(x: UQ(2, 0), y: UQ(3, 0), ctx) -> UQ:
    return x + y


# The result can be negative
@Test(InfeasibleError)
def spec_infeasible_subtraction(x: UQ(3, 0), y: UQ(1, 0), ctx) -> UQ:
    return x - y


# Result is positive in this case, UQ is enough
@Test()
def spec_feasible_subtraction(x: UQ(3, 0), y: UQ(1, 0), ctx) -> UQ:
    ctx.assume(x > ctx.one())
    return x - y


# UQ is enough
@Test()
def spec_feasible_subtraction2(x: UQ(3, 0), y: UQ(1, 0), ctx) -> UQ:
    ctx.assume((x - y) >= ctx.real_val(1))
    return x - y


# Everything is good here
@Test()
def spec2(x: UQ(2, 0), y: Q(3, 0), ctx) -> Q:
    return x + y


# x does not have a clear precision
@Test(MissingError)
def spec3(x: UQ, y: Q(3, 0), ctx) -> Q:
    return x + y


# infeasible
@Test(InfeasibleError)
def spec4(x: UQ(2, 0), y: Q(3, 0), ctx) -> UQ(3, 0):
    return x + y


@Test()
def spec5(x: UQ(2, 0), ctx) -> UQ:
    return x + ctx.one()


# sign extension, msb is not going to be used
@Test()
def spec6(x: Q(2, 0), ctx) -> Q(4, 0):
    return x + ctx.one()


@Test()
def spec7(x: Q(3, 0), y: Bool(), ctx) -> Q(3,0):
    return If(y.eq(ctx.false()), x, ctx.one())


# some undeclared variables in spec
@Test(MissingError)
def spec8(x: Q(3, 0), ctx) -> Q:
    y = ctx.fresh_real("x")
    return x + ctx.real_val(2) + y


# precision for 2.5 is not known
@Test(NotImplementedError)
def spec9(x: Q(3, 0), ctx) -> Q:
    return x + ctx.real_val(2.5)


# conditional selection
@Test()
def spec10(sel: Bool(), in1: UQ(2, 0), in0: UQ(2, 0), ctx) -> UQ(2, 0):
    return If(sel, in1, in0)


# infeasible
@Test(InfeasibleError)
def spec11(x: UQ(2, 0), ctx) -> UQ(2, 0):
    return x + ctx.one()


# boolean type but spec returns real expression
@Test(TypeError)
def spec12(x: UQ(2, 0), ctx) -> Bool():
    return x + ctx.one()


# float32 as output is not supported currently
@Test(NotImplementedError)
def spec13(x: UQ(2, 0), ctx) -> Float32():
    return x + ctx.one()


# conversion can be composed with numeric components
@Test()
def spec_bool_to_uq_then_add(x: Bool(), y: UQ(1, 0), ctx) -> UQ(4, 0):
    return If(x, ctx.one(), ctx.zero()) + y
