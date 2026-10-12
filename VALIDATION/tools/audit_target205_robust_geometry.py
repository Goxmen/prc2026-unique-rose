"""Reuse the independent SQL auditor against the updated scored basis."""
import types

import audit_robust_scored_geometry as original
import target205_robust_geometry as experiment


def main():
    source=original.audit
    namespace=dict(source.__globals__,experiment=experiment,__file__=__file__)
    fn=types.FunctionType(source.__code__,namespace,'updated_basis_audit',source.__defaults__,source.__closure__)
    assert fn.__code__ is source.__code__
    fn()
    path=experiment.OUT/'audit.json';value=experiment.read(path)
    value['reused_auditor']=experiment.art(original.__file__)
    value['identical_independent_sql_audit_bytecode']=True
    original.write_json(path,value)


if __name__=='__main__':
    main()
