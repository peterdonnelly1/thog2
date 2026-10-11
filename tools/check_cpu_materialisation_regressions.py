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
RENAMED_BASELINE_TESTS = {
    "tests.test_premat::test_public_cli_exposes_exactly_the_seventeen_premat_options":
        "tests.test_premat::test_public_cli_exposes_existing_options_and_nine_cpu_controls",
}
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
    print(f"Starting {label} complete pytest suite; live output: {output / (label + '.log')}", flush=True)
    with (output/f"{label}.log").open("w") as log:
        run=subprocess.run(args,cwd=folder,env=environment,stdout=log,stderr=subprocess.STDOUT,text=True)
    captured=(output/f"{label}.log").read_text()
    print(f"{label} pytest exit={run.returncode}\n"+captured[-6000:],flush=True)
    if run.returncode not in (0,1):infrastructure.append(label+": pytest exit "+str(run.returncode))
    failures=[]
    failure_counts={}
    test_ids=set()
    count=skipped=0
    if xml.exists():
        tree=ET.parse(xml)
        for case in tree.iter("testcase"):
            count+=1
            identity=case.attrib.get("classname","")+"::"+case.attrib.get("name","")
            test_ids.add(identity)
            if case.find("skipped") is not None:skipped+=1
            case_failures=[child for child in case if child.tag in ("failure","error")]
            if case_failures:
                failures.append(identity)
                failure_counts[identity]=failure_counts.get(identity,0)+len(case_failures)
    if run.returncode and not failures:
        failures=["pytest collection/infrastructure: "+captured[-4000:]]
    results[label]={"tests":count,"skipped":skipped,"failures":failures,"failure_counts":failure_counts,"test_ids":sorted(test_ids)}
introduced=sorted(set(results["candidate"]["failures"])-set(results["base"]["failures"]))
resolved=sorted(set(results["base"]["failures"])-set(results["candidate"]["failures"]))
increased={identity:count-results["base"]["failure_counts"][identity]
           for identity,count in results["candidate"]["failure_counts"].items()
           if identity in results["base"]["failure_counts"] and count>results["base"]["failure_counts"][identity]}
expected_tests={RENAMED_BASELINE_TESTS.get(identity,identity) for identity in results["base"]["test_ids"]}
missing_tests=sorted(expected_tests-set(results["candidate"]["test_ids"]))
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
payload={"base_commit":"e29e32db916fc5ceb87a089d057fb78456c1a98a","pytest":results,"introduced_failures":introduced,"resolved_failures":resolved,"increased_existing_failure_counts":increased,"missing_baseline_tests":missing_tests,"renamed_baseline_tests":RENAMED_BASELINE_TESTS,"javascript":js,"introduced_javascript_failures":js_introduced,"excluded_legacy_browser_tests":list(EXCLUDED)}
(output/"comparison.json").write_text(json.dumps(payload,indent=2)+"\n")
print(json.dumps({key:payload[key] for key in ("introduced_failures","resolved_failures","increased_existing_failure_counts","missing_baseline_tests","introduced_javascript_failures")},indent=2),flush=True)
print("Infrastructure failures: "+repr(infrastructure),flush=True)
raise SystemExit(bool(introduced or increased or missing_tests or js_introduced or infrastructure))
# ^^^ THOG
