// vvv THOG exercise conditional Runner controls with native typing and saved Recipes
"use strict";
const assert=require("node:assert/strict"),{chromium,firefox}=require("playwright");
const address=process.env.INSTRA_TEST_URL || "http://127.0.0.1:8765";

async function check(browser_type) {
  const browser=await browser_type.launch({headless:true,...(browser_type===chromium && process.env.INSTRA_CHROMIUM_PATH ? {
    executablePath:process.env.INSTRA_CHROMIUM_PATH,args:["--no-sandbox","--disable-dev-shm-usage"],
  } : {}),...(browser_type===firefox ? {env:{...process.env,MOZ_DISABLE_CONTENT_SANDBOX:"1"}} : {})});
  try {
    const page=await browser.newPage({viewport:{width:1600,height:1000}}),errors=[];
    page.setDefaultTimeout(20000);
    page.on("pageerror",error=>errors.push(error.message));
    await page.goto(address+"/runs/fixture_00");
    await page.locator("#runner_nav").click();
    await page.waitForSelector(".runner-categories");
    const field=key=>page.locator(`input[data-runner-field="${key}"]`);
    const select_category=name=>page.locator(`.runner-categories button[data-runner-category="${name}"]`).click();
    await select_category("Premat");

    const cpu_keys=await page.locator('input[data-runner-field^="--premat_cpu_"]').evaluateAll(nodes=>nodes.map(node=>node.dataset.runnerField));
    assert.ok(cpu_keys.length>=8);
    for(const key of cpu_keys) assert.equal(await field(key).isDisabled(),true,`${key} must start inactive for the default GPU provider`);
    assert.equal(await field("--premat_timing").isEnabled(),true);

    await field("--premat").pressSequentially("enabled");
    await field("--premat_materialisation_device").pressSequentially("cpu_and_gpu");
    const preparation=field("--premat_cpu_preparation");
    assert.equal(await preparation.isEnabled(),true);
    await preparation.pressSequentially("e");
    assert.equal(await preparation.inputValue(),"e");
    assert.equal(await preparation.isEnabled(),true,"an incomplete enum value must stay editable");
    assert.equal(await preparation.getAttribute("aria-invalid"),"true");
    assert.equal(await preparation.evaluate(node=>document.activeElement===node),true);
    await preparation.pressSequentially("ager");
    assert.equal(await preparation.inputValue(),"eager");
    assert.equal(await preparation.getAttribute("aria-invalid"),"false");
    assert.equal(await field("--premat_timing").isDisabled(),true);
    assert.equal(await field("--premat_cpu_transfer_lead_ms").isDisabled(),true);

    await field("--premat_cpu_transfer_timing").pressSequentially("predicted_gemm_start");
    assert.equal(await field("--premat_cpu_transfer_lead_ms").isEnabled(),true);
    await field("--premat_cpu_transfer_lead_ms").pressSequentially("3.5");
    await preparation.press("End");
    await preparation.pressSequentially(", scheduled");
    assert.equal(await preparation.inputValue(),"eager, scheduled");
    assert.equal(await preparation.getAttribute("aria-invalid"),"false");
    await field("--premat_cpu_workers").pressSequentially("12");
    assert.equal(await field("--premat_cpu_workers").inputValue(),"12");
    await field("--premat_cpu_workers").fill("1");

    await select_category("Frequently Used");
    await select_category("Premat");
    assert.equal(await preparation.inputValue(),"eager, scheduled");
    assert.equal(await preparation.isEnabled(),true);
    assert.equal(await field("--premat_timing").isDisabled(),true);
    assert.equal(await field("--premat_cpu_transfer_lead_ms").isEnabled(),true);

    await field("--premat_materialisation_device").fill("gpu");
    await select_category("Frequently Used");
    await select_category("Premat");
    for(const key of cpu_keys) assert.equal(await field(key).isDisabled(),true,`${key} must remain inactive after switching tabs`);
    assert.equal(await preparation.inputValue(),"eager, scheduled","inactive CPU requests must be retained");
    assert.equal(await preparation.getAttribute("aria-invalid"),"false");
    assert.equal(await field("--premat_timing").isEnabled(),true);

    const search=page.locator("#runner_parameter_search");
    await search.fill("cpu_preparation");
    assert.equal(await preparation.isDisabled(),true,"search results must apply the same provider rules");
    await search.fill("");
    await field("--premat_materialisation_device").fill("gpu, cpu_and_gpu");
    assert.equal(await preparation.isEnabled(),true,"a mixed-provider sweep must allow CPU requests");
    await field("--premat_cpu_transfer_timing").fill("as_the_code_flies");
    assert.equal(await field("--premat_cpu_transfer_lead_ms").isDisabled(),true);
    await select_category("Frequently Used");
    await select_category("Premat");
    assert.equal(await field("--premat_cpu_transfer_lead_ms").isDisabled(),true);
    assert.equal(await field("--premat_cpu_transfer_lead_ms").inputValue(),"3.5");

    await field("--premat_materialisation_device").fill("cpu_and_gpu");
    await field("--premat_cpu_transfer_timing").fill("predicted_gemm_start");
    await field("--premat_cpu_transfer_lead_ms").fill("0");
    await preparation.fill("eager");
    const label=`CPU typing ${browser_type.name()} ${Date.now()}`;
    await field("Recipe label").fill(label);
    const saved_response=page.waitForResponse(response=>response.url().endsWith("/api/runner/action") && response.request().postDataJSON()?.action==="save");
    await page.locator(".runner-actions button",{hasText:/^Save$/}).click();
    const response=await saved_response;
    assert.equal(response.ok(),true,await response.text());
    const state=await (await page.request.get(address+"/api/runner")).json();
    const saved=state.recipes.find(item=>item.recipe.label===label);
    assert.ok(saved,"the completed fields must reach Recipe persistence");
    assert.deepEqual(saved.recipe.parameters["--premat_cpu_preparation"],["eager"]);
    assert.deepEqual(saved.recipe.parameters["--premat_materialisation_device"],["cpu_and_gpu"]);
    assert.deepEqual(errors,[]);
    console.log(JSON.stringify({browser:browser_type.name(),cpu_controls:cpu_keys.length,typing:true,tabs:true,search:true,sweep:true,saved:true,page_errors:errors}));
  } finally {await browser.close();}
}

(async()=>{
  for(const name of (process.env.INSTRA_TEST_BROWSERS || "chromium,firefox").split(",")) {
    const browser_type={chromium,firefox}[name];assert.ok(browser_type,name);await check(browser_type);
  }
})().catch(error=>{console.error(error.stack);process.exitCode=1;});
// ^^^ THOG
