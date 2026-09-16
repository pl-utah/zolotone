import typing as tp

from .nodes import _build_spec_contract


def Autogenerate(name: str, spec: tp.Callable[..., tp.Any]):
    contract = _build_spec_contract(name, spec)
    print(contract)