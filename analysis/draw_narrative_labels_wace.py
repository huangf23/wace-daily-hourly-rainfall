"""Change panel wording only, preserving archived calculations and figure sizes."""
from pathlib import Path
import json
R=Path(__file__).resolve().parent
dest=R/'manuscript_elsevier/figures_wace/narrative_revision_20261002';dest.mkdir(parents=True,exist_ok=True)
def module(name,replacements):
 src=(R/name).read_text(encoding='utf-8')
 for a,b in replacements:
  assert a in src,a
  src=src.replace(a,b)
 ns={'__name__':'narrative_labels','__file__':str(R/name)}
 exec(compile(src,str(R/name),'exec'),ns)
 ns['DEST']=dest
 return ns
a=module('draw_focused_revision_wace.py',[
 ('Known-parameter discrepancy','Population transfer error'),
 ('Forty-year estimation','Error with 40-year samples'),
 ('Aggregate percentile intervals','Nominal 95% percentile intervals'),
 ('Interval displacement, member 01','Bootstrap displacement, member 01'),
 ('These intervals have not been validated for nominal coverage.','Coverage controls in Fig.~S31 show undercoverage for some local contrasts; model intervals describe conditional sampling sensitivity.'),
])
a['controls']();a['uncertainty']()
b=module('draw_submission_checks_wace.py',[
 ("'Daily change signal'","'Signed daily change'"),
 ("'Historical tail mismatch'","'Historical tail-growth mismatch'"),
 ("'Historical seasonal concentration'","'Historical AMS seasonality'"),
])
b['conditions']()
(dest/'captions.json').write_text(json.dumps(a['CAP']|b['CAP'],indent=2),encoding='utf-8')
print('Three figures regenerated from unchanged data; panel labels only.')
