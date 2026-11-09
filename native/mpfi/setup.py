from setuptools import Extension, setup

setup(
    name="zolotone-mpfi-bridge",
    version="0.1.0",
    ext_modules=[Extension("_zolotone_mpfi", ["bridge.c"], libraries=["mpfi", "mpfr", "gmp"])],
)
