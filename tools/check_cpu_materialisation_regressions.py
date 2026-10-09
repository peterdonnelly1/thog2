# vvv THOG compare complete CPU-capable pytest and static Instra suites against the exact enhancement base
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

EXCLUDED = (
    "tests/test_instra_firefox_acceptance.py", "tests/test_instra_firefox_heatmap_diagnostics.py",
    "tests/test_instra_firefox_maximize_history.py", "tests/test_instra_firefox_weight_controls.py",
    "tests/test_instra_firefox_workspace_range.py",
)
root=Path(__file__).resolve().parents[1]
base=Path(sys.argv[1]).resolve()
output=root/"evidence"/"cpu_materialisation_ci"
output.mkdir(parents=True,exist_ok=True)
environment=dict(os.environ, PYTHONPATH=".", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
results={}
infrastructure=[]
for label, folder in (("base",base),("candidate",root)):
    xml=output/f"{label}.xml"
    args=[sys.executable,"-m","pytest","-q","tests","--timeout=60","--junitxml="+str(xml),*[f"--ignore={p}" for p in EXCLUDED]]
    run=subprocess.run(args,cwd=folder,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (output/f"{label}.log").write_text(run.stdout)
    print(f"{label} pytest exit={run.returncode}\n"+run.stdout[-6000:],flush=True)
    if run.returncode not in (0,1):infrastructure.append(label+": pytest exit "+str(run.returncode))
    failures=[]
    count=skipped=0
    if xml.exists():
        tree=ET.parse(xml)
        for case in tree.iter("testcase"):
            count+=1
            if case.find("skipped") is not None:skipped+=1
            if case.find("failure") is not None or case.find("error") is not None:
                failures.append(case.attrib.get("classname","")+"::"+case.attrib.get("name",""))
    if run.returncode and not failures:
        failures=["pytest collection/infrastructure: "+run.stdout[-4000:]]
    results[label]={"tests":count,"skipped":skipped,"failures":failures}
introduced=sorted(set(results["candidate"]["failures"])-set(results["base"]["failures"]))
resolved=sorted(set(results["base"]["failures"])-set(results["candidate"]["failures"]))
js={}
for label,folder in (("base",base),("candidate",root)):
    js[label]={}
    for script in sorted((folder/"tests").glob("instra*regression*.js")):
        text=script.read_text()
        if "playwright" in text or "puppeteer" in text:continue
        run=subprocess.run(["node",str(script)],cwd=folder,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=90)
        js[label][script.name]={"exit":run.returncode,"output":run.stdout[-2500:]}
        print(f"{label} {script.name}: "+("PASS" if not run.returncode else run.stdout[-1200:]),flush=True)
js_introduced=[name for name,item in js["candidate"].items() if item["exit"] and not js["base"].get(name,{"exit":0})["exit"]]
payload={"base_commit":"e29e32db916fc5ceb87a089d057fb78456c1a98a","pytest":results,"introduced_failures":introduced,"resolved_failures":resolved,"javascript":js,"introduced_javascript_failures":js_introduced,"excluded_legacy_browser_tests":list(EXCLUDED)}
(output/"comparison.json").write_text(json.dumps(payload,indent=2)+"\n")
print(json.dumps({key:payload[key] for key in ("introduced_failures","resolved_failures","introduced_javascript_failures")},indent=2),flush=True)
print("Infrastructure failures: "+repr(infrastructure),flush=True)
raise SystemExit(bool(introduced or js_introduced or infrastructure))
# ^^^ THOG
