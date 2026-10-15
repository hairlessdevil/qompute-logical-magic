"""Build summary reports from archived results; never submit or rerun an experiment.

Run --figures with quantum-jax, then --reports with a Python containing reportlab.
All numerical tables are populated from the saved JSON analysis, not retyped.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
FIG = OUT / "figures"
PDF = ROOT / "output/pdf"
SIM = ROOT / "results/noisy_aachen"
HW = ROOT / "results/hardware_b2ed"
INPUTS = {}


def read(path):
    path = Path(path)
    INPUTS[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


base = read(SIM / "noisy_results.json")
original = [r for r in base["rows"] if r["strategy"].startswith("B2")]
stage = read(SIM / "syndrome_delay_ablation/results.json")["rows"]
residual = read(SIM / "syndrome_delay_ablation/residual_codespace_loss.json")["rows"]
finite = [r for r in read(SIM / "readout_and_fidelity.json")["rows"] if r["strategy"].startswith("B2")]
filtering = read(SIM / "syndrome_filtering.json")["rows"]
dd = read(SIM / "dd_full_noise/results.json")["rows"]
timing = read(SIM / "timelines/timing_summary.json")["rows"]
mit = read(HW / "readout_mitigation_20261014/readout_mitigation.json")
bundles = [HW / "four_pilot_20261014", HW / "four_xy4_20261014"]
manifests = [read(p / "manifest.json") for p in bundles]
analyses = [read(p / "analysis_extended.json") for p in bundles]
replays = [read(p / "calibration_replay.json") for p in bundles]
for p in bundles:
    read(p / "submission.json")


def name(r):
    return ("Four" if r["code"] == "four" else "Five") + " / " + r["strategy"]


def pct(v, n=2):
    return f"{100*v:.{n}f}%"


def ci(vals, n=2):
    return f"[{100*vals[0]:.{n}f}, {100*vals[1]:.{n}f}]"


def pp(v, n=2):
    return f"{100*v:+.{n}f}"


def make_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch, Rectangle
    from draw_timelines import AachenPatch, draw_timeline, restore_schedule
    from qiskit import qpy
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white",
                         "savefig.facecolor": "white", "font.family": "DejaVu Sans"})
    colors = ["#284b70", "#b17820", "#208a88", "#8a6fb2"]

    # Redraw the frozen QPY schedules with uncropped figure bounds. This only
    # restores scheduling metadata; no circuit synthesis or simulation is run.
    backend = AachenPatch(base["backend"]["physical_patch"])
    for filename, info in base["backend"]["snapshot_files"].items():
        assert info["sha256"] == backend.provenance()["snapshot_files"][filename]["sha256"]
    for r, saved_timing in zip(original, timing):
        stem = f"{r['code']}_{r['strategy']}"
        source = SIM / "circuits" / f"{stem}.qpy"
        assert hashlib.sha256(source.read_bytes()).hexdigest() == saved_timing["source_qpy_sha256"]
        with source.open("rb") as f:
            circuit, record = restore_schedule(qpy.load(f)[0], backend, r["resources"])
        assert record == saved_timing["original"]
        draw_timeline(circuit, backend, record,
                      r["resources"]["final_data_local_wires"],
                      r["resources"]["syndrome_local_wires_in_bit_order"],
                      name(r) + " - original Aachen schedule",
                      FIG / f"simulation_{stem}_timeline")

    def save(fig, stem):
        fig.savefig(FIG / f"{stem}.png", dpi=220, bbox_inches="tight")
        fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 2.8), layout="constrained")
    y = np.arange(3)
    outside = np.array([100*(1-r["full"]["codespace_population"]) for r in original])
    inside = np.array([100*(r["full"]["codespace_population"]-r["full"]["fidelity"]) for r in original])
    ax.barh(y, outside, color=colors[0], label="Population outside code")
    ax.barh(y, inside, left=outside, color=colors[1], label="Within-code distortion, weighted")
    for i, (o, d) in enumerate(zip(outside, inside)):
        ax.text(o+d+.17, i, f"{o+d:.2f}% total", va="center")
    ax.set(yticks=y, yticklabels=[name(r) for r in original], xlabel="Contribution to accepted-state infidelity (percentage points)", xlim=(0, 18))
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(.52, 1.25), ncols=2, frameon=False, fontsize=9)
    save(fig, "simulation_infidelity_split")

    fig, ax = plt.subplots(figsize=(9, 2.9), layout="constrained")
    for i, r in enumerate(stage):
        t = r["timing"]
        durations = [t["extraction_start_us"], t["readout_start_us"]-t["extraction_start_us"], t["delivery_us"]-t["readout_start_us"]]
        start = 0
        for j, d in enumerate(durations):
            ax.barh(i, d, left=start, color=["#b6cadb", "#208a88", "#e9bc6d"][j], height=.55)
            if d>.3: ax.text(start+d/2, i, f"{d:.3f}", ha="center", va="center", fontsize=9)
            start += d
        ax.text(start+.1, i, f"{start:.3f}", va="center", fontsize=9)
    ax.set(yticks=range(3), yticklabels=[name(r) for r in stage], xlabel="Time from circuit start (µs)", xlim=(0, 13))
    ax.invert_yaxis()
    ax.legend(handles=[Patch(color=c, label=l) for c,l in zip(["#b6cadb", "#208a88", "#e9bc6d"], ["Preparation", "Syndrome mapping", "Readout / data wait"])], loc="upper center", bbox_to_anchor=(.5,1.25), ncols=3, frameon=False, fontsize=9)
    save(fig, "simulation_staged_timing")

    fig, ax = plt.subplots(figsize=(9, 3), layout="constrained")
    modes = ["full", "readout_delay_off", "extraction_delay_off", "all_delay_off"]
    labels = ["Full noise", "Readout-wait idle off", "Extraction + readout idle off", "All idle off"]
    for j, (mode, label) in enumerate(zip(modes, labels)):
        vals = [100*r["modes"][mode]["fidelity"] for r in stage]
        ax.plot(vals, np.arange(3)+(j-1.5)*.13, "o", color=colors[j], label=label, markersize=7)
    ax.set(yticks=range(3), yticklabels=[name(r) for r in stage], xlabel="Full accepted-state fidelity (%)", xlim=(80,100))
    ax.invert_yaxis(); ax.grid(axis="x", alpha=.2)
    ax.legend(loc="upper center", bbox_to_anchor=(.5,1.35), ncols=2, frameon=False, fontsize=9)
    save(fig, "simulation_delay_comparison")

    fig, ax = plt.subplots(figsize=(9, 3.1), layout="constrained")
    for j, (mode, label) in enumerate([("raw_recorded_zero", "Raw measured score"), ("data_corrected_recorded_zero", "Data-readout corrected (primary)")]):
        y = np.arange(2)+(j-.5)*.20
        vals = np.array([100*r["fidelity"][mode]["estimate"] for r in mit["runs"]])
        err = np.array([[100*r["fidelity"][mode]["normal_95_interval"][k] for r in mit["runs"]] for k in [0,1]])
        ax.errorbar(vals,y,xerr=[vals-err[0],err[1]-vals],fmt="o",capsize=4,color=colors[j*2],label=label,markersize=7)
        for x,yy in zip(vals,y): ax.text(x,yy+.10,f"{x:.2f}%",ha="center",va="top",fontsize=10)
    ax.set(yticks=[0,1],yticklabels=["No DD", "XY4"],xlabel="Accepted-state target fidelity estimate (%)",xlim=(65,89),ylim=(1.5,-.5))
    ax.grid(axis="x",alpha=.2)
    ax.legend(loc="upper center",bbox_to_anchor=(.5,1.25),ncols=2,frameon=False,fontsize=9)
    save(fig, "hardware_fidelity")

    fig, ax = plt.subplots(figsize=(9, 3.4), layout="constrained")
    selected = [w for w in manifests[1]["resources"]["dd"]["windows"] if w["start_dt"]==710]
    selected.sort(key=lambda w: [5,3,2,16].index(w["physical_qubit"]))
    for i,w in enumerate(selected):
        ax.add_patch(Rectangle((2.840,i-.22),2.180,.44,color="#e2e8ee"))
        for j,t in enumerate(w["pulse_starts_dt"]):
            x=t*.004; col=colors[0] if j%2==0 else colors[2]
            ax.add_patch(Rectangle((x,i-.22),.032,.44,color=col))
            ax.text(x+.016,i-.28,"X" if j%2==0 else "Y",ha="center",va="bottom",fontsize=9)
    ax.barh(4,2.180,left=2.840,color="#e9bc6d",height=.44)
    ax.text(3.93,4,"Ancilla measurement: 2.180 µs",ha="center",va="center",fontsize=9)
    ax.axvline(2.840,color="#667788",ls="--",lw=.8);ax.axvline(5.020,color="#667788",ls="--",lw=.8)
    ax.set(yticks=range(5),yticklabels=[f"data q{i} / p{w['physical_qubit']}" for i,w in enumerate(selected)]+["3 syndrome ancillas"],xlabel="Time from circuit start (µs)",xlim=(2.75,5.1),ylim=(4.6,-.8))
    ax.set_title("Hardware readout window: XY4 on all four output data qubits",fontsize=12,pad=13)
    save(fig,"hardware_xy4_readout_timing")


# Each block is shared by the Markdown and PDF renderers.
def P(text): return ("p",text.replace("〈", "⟨").replace("〉", "⟩"))
def H(text): return ("h",text)
def T(headers, rows, widths=None): return ("table", headers, rows, widths)
def I(path, caption, max_height=250): return ("image",Path(path),caption,max_height)
def N(text): return ("note",text.replace("〈", "⟨").replace("〉", "⟩"))


def simulation_pages():
    full_rows=[]; split_rows=[]
    for r in original:
        f=r["full"]; full_rows.append([name(r),pct(f["acceptance"]),pct(f["fidelity"]),pct(f["codespace_population"]),pct(f["fidelity_conditioned_on_codespace_diagnostic"])])
        split_rows.append([name(r),pct(1-f["fidelity"],3),pct(1-f["codespace_population"],3),pct(f["codespace_population"]-f["fidelity"],3)])
    pages=[[
        H("Findings and model scope"),
        P("The dominant visible defect is population outside the code subspace. Within that subspace, accepted states retain 99.09-99.76% target fidelity. Removing syndrome-extraction idle noise produces substantial improvements; residual loss is largely associated with noisy extraction gates, especially CZ channels. Syndrome assignment errors are a smaller effect in this model."),
        P("These are local Aer results from the packaged IBM Aachen calibration dated 17 April 2026. No manually inserted Pauli faults are used. Every B2 state is conditioned only on the recorded all-zero syndrome, with no adaptive correction or extra code-subspace filtering."),
        T(["Protocol","Accept","Full F","In code","F within code"],full_rows,[1.45,.7,.7,.7,1.05]),
        P("Four / B2_ED prepares the [[4,1,2]] encoded H state, cos(π/8)|0L〉 + sin(π/8)|1L〉. Five / B2_ED uses the [[5,1,3]] code and a tetrahedral magic target with logical Bloch vector (1,1,1)/√3. Five / B2_BK projects five antipodal tetrahedral inputs; its ideal acceptance is 1/6."),
        H("Included noise and timing"),
        P("Native gates carry calibrated depolarizing and gate-duration thermal-relaxation channels. Every scheduled idle has T1/T2 relaxation applied once, at zero temperature, with effective T2 capped at 2T1. Independent symmetric syndrome-readout assignment errors come from the BackendV2 Target averages. Final data-qubit readout errors are also present in the finite-shot quality measurements."),
        P("ALAP scheduling uses dt = 4 ns on physical patch [3, 2, 4, 16, 1, 5, 23, 0, 6]. Data remain noisy during the entire 2.584 µs terminal syndrome-readout wait. Initial qubits are ideal |0〉; preparation gates are noisy. The standard Aer measurement model adds assignment error but no extra ancilla thermal channel over its own measurement duration."),
        N("Not modeled: coherent or correlated errors, slow frequency drift, crosstalk, higher-level transmon leakage, explicit reset errors, and additional classical-processing or storage latency. 'Outside code' refers to the modeled qubit Hilbert space, not occupation of a transmon's |2〉 level."),
        P("Metrics: p_code = Tr(P_code ρ_A), F = 〈ψL|ρ_A|ψL〉, and F_code = F / p_code. Here ρ_A is the delivered accepted density matrix. All numbers in the table are exact model quantities, without shot uncertainty. [S1]"),
        I(FIG/"simulation_infidelity_split.png","Accepted-state infidelity split into outside-code population and weighted within-code distortion. See the exact identity and numerical table on the residual-loss page.",125),
    ],[
        H("Original circuits: native instruction timing"),
        P("The Qiskit timeline plots below show the complete preparation and syndrome circuit, ending when the accepted data are delivered. Gray regions are idles. The final amber region is ancilla readout, during which the data wait and relax."),
        I(FIG/"simulation_four_B2_ED_timeline.png","Figure 1. Four / B2_ED: 2.840 µs of preparation and syndrome mapping, followed by 2.584 µs readout; total 5.424 µs. [S2]",245),
        I(FIG/"simulation_five_B2_ED_timeline.png","Figure 2. Five / B2_ED: 7.812 µs before readout; total 10.396 µs. Labels D/S identify output data/syndrome roles; routing can move states. [S2]",245),
        N("These are Qiskit instruction-duration diagrams, not measured analog pulse envelopes. The saved backend snapshot does not provide waveform shapes or amplitudes. The original compiler can mix preparation and extraction operations, so no precise preparation/extraction boundary is inferred for these circuits."),
    ],[
        H("Timing and the matched phase-ablation circuits"),
        I(FIG/"simulation_five_B2_BK_timeline.png","Figure 3. Original Five / B2_BK: 3.640 µs before readout, 2.584 µs readout; total 6.224 µs. [S2]",225),
        P("To switch off noise in a specified phase, we used separately compiled diagnostic circuits with preserved stage boundaries. Their routing and gate counts can differ from the original circuits. Every phase-ablation result below therefore uses its own recomputed full-noise baseline and the same frozen circuit across controls."),
        I(FIG/"simulation_staged_timing.png","Figure 4. Stage-separated circuits used for phase ablations. Numbers are wall-clock durations in µs, not sums of idle time over qubits. [S3]",150),
        T(["Stage-separated protocol","Prep","Mapping","Readout","Total extraction*","Total"],[[name(r),f"{r['timing']['extraction_start_us']:.3f}",f"{r['timing']['readout_start_us']-r['timing']['extraction_start_us']:.3f}","2.584",f"{r['timing']['delivery_us']-r['timing']['extraction_start_us']:.3f}",f"{r['timing']['delivery_us']:.3f}"]for r in stage],[1.75,.5,.65,.65,.95,.6]),
        N("* Extraction includes syndrome mapping and readout. Turning off an idle-noise channel does not shorten this schedule. Extraction controls cover all physical wires from the end of preparation through completed readout."),
    ],[
        H("Certain noise on/off: delay relaxation"),
        P("Only the indicated scheduled Delay channels are suppressed. Native gates, routing, gate-duration relaxation, measurement duration, and assignment error stay fixed. This is a hypothetical channel-removal test, not a claimed hardware implementation."),
        I(FIG/"simulation_delay_comparison.png","Figure 5. Matched phase controls: full target fidelity of the delivered accepted state. [S3]",160),
        T(["Protocol / control","Accept","Full F","In code","F within code"],[[name(r)+" / "+label,pct(m["acceptance"],3),pct(m["fidelity"],3),pct(m["codespace_population"],3),pct(m["fidelity_conditioned_on_codespace_diagnostic"],3)]for r in stage for key,label in [("full","Full"),("readout_delay_off","Readout idle off"),("extraction_delay_off","Extraction idle off"),("all_delay_off","All idle off")]for m in [r["modes"][key]]],[2.5,.78,.78,.78,.9]),
        P("Suppressing extraction idles raises full fidelity by "+", ".join(f"{100*(r['modes']['extraction_delay_off']['fidelity']-r['modes']['full']['fidelity']):.2f} points"for r in stage)+" for Four ED, Five ED and Five BK. Suppressing only the last readout wait already raises fidelity by "+", ".join(f"{100*(r['modes']['readout_delay_off']['fidelity']-r['modes']['full']['fidelity']):.2f} points"for r in stage)+". That final wait does not change acceptance because it acts on data after syndrome mapping is complete."),
        N("These differences identify the benefit of specific interventions. They are not additive shares of a unique error budget: disabling noise can alter syndrome probabilities and the accepted ensemble. The dominant source is protocol-dependent; the five-qubit circuits retain a substantial gate-noise contribution."),
    ],[
        H("Where the remaining code-subspace loss comes from"),
        P("The following controls all start with extraction idle noise OFF in the same stage-separated circuits. Each additional removal is compared to that baseline; the rows are not sequential unless explicitly combined. Numbers are accepted population outside the code subspace. [S4]"),
        T(["Additional idealization","Four ED","Five ED","Five BK"],[[label]+[pct(r["modes"][key]["outside_code_population"],4)for r in residual]for key,label in [("baseline","None: extraction idle off"),("perfect_readout","Perfect syndrome assignment"),("perfect_extraction_cz","Perfect extraction CZ channels"),("perfect_extraction_gates","Perfect all extraction gate channels"),("perfect_extraction_gates_and_readout","Perfect extraction gates + assignment"),("perfect_preparation","Perfect preparation gates + idles")]],[2.9,.8,.8,.8]),
        T(["Full target fidelity under control","Four ED","Five ED","Five BK"],[[label]+[pct(r["modes"][key]["fidelity"],3)for r in residual]for key,label in [("baseline","Extraction idle off"),("perfect_readout","Also perfect syndrome assignment"),("perfect_extraction_cz","Also perfect extraction CZ"),("perfect_extraction_gates","Also perfect extraction gates"),("perfect_extraction_gates_and_readout","Also perfect extraction gates + assignment")]],[2.9,.8,.8,.8]),
        P("Removing the extraction CZ channels eliminates most of the residual outside-code population. Those channels contain both depolarizing error and relaxation over the CZ duration; this study does not separate the two. Perfect preparation alone gives little improvement to the residual code-subspace loss."),
        P("Why does the syndrome miss these faults? A data error can occur after the relevant stabilizer has already been mapped to an ancilla. The old ancilla record then describes an earlier state. Later checks may commute with that error, and the earlier check is not repeated. Faulty ancilla interactions can also corrupt a syndrome or propagate errors to data. Perfect assignment cannot repair this quantum-circuit failure."),
        H("A valid additive decomposition of the final state"),
        P("For the original full-noise circuits, 1 - F = (1 - p_code) + (p_code - F). The first term is outside-code population; the second is weighted distortion inside the code. Unlike the on/off comparisons, these two terms do add exactly."),
        T(["Original protocol","Total infidelity","Outside code","Within-code term"],split_rows,[1.7,1.1,1.1,1.2]),
        N("With ideal extraction gates, ideal assignment and extraction idles suppressed, the accepted state returns to the code subspace to numerical precision. Its logical state need not be perfect: the remaining full fidelities are 99.839%, 99.182% and 99.9999%."),
    ],[
        H("Readout error and finite-shot state quality"),
        P("The first table removes only syndrome assignment error in the original circuits. Measurement time and the noisy data wait remain. This isolates mislabeling of accept/reject from physical relaxation during readout. [S5]"),
        T(["Original protocol","Accept: assignment off → on","Full F: off → on","False-syndrome share*"],[[name(r),pct(r["without_syndrome_readout_error"]["acceptance"],3)+" → "+pct(r["with_syndrome_readout_error"]["acceptance"],3),pct(r["without_syndrome_readout_error"]["fidelity"],3)+" → "+pct(r["with_syndrome_readout_error"]["fidelity"],3),pct(r["with_syndrome_readout_error"]["accepted_fraction_from_nonzero_true_syndrome"],4)]for r in finite],[1.5,1.65,1.55,1.05]),
        N("* Fraction of accepted runs whose true ancilla syndrome was nonzero. This is not the entire outside-code population: faults during and after extraction can evade an otherwise correctly read syndrome."),
        H("Estimating fidelity from measurements"),
        P("Expand the full physical target projector as Σ c_P P. Measure each nonidentity Pauli on fresh copies; retain recorded-zero shots in each setting and combine their parities. Four-qubit targets use 23 settings and five-qubit targets 63, with 8,192 attempted shots per setting. Analysis rotations and final data readout are noisy."),
        T(["Protocol","Exact density F","Raw estimate","Data corrected","95% sampling CI"],[[name(r),pct(r["exact_preparation_fidelity"],3),pct(r["finite_shot"]["raw"]["fidelity_estimate"],3),pct(r["finite_shot"]["data_readout_mitigated"]["fidelity_estimate"],3),ci(r["finite_shot"]["data_readout_mitigated"]["normal_approx_95_interval"],3)]for r in finite],[1.45,1,1,1,1.3]),
        P("Correction inverts final data assignment errors using the exact matrices of the simulation. It leaves the recorded syndrome selection unchanged. Physical qubit 2 has an 8.67% assignment error in this snapshot, causing much of the raw-estimate bias. The corrected estimates agree with the density-matrix values within sampling uncertainty; small residual bias from noisy analysis gates/delays remains. The intervals do not include model uncertainty."),
        H("How much bad population does postselection remove?"),
        T(["Original protocol","Outside code rejected / all outside code","Accepted F","All-outcomes F"],[[name(r),pct(r["fraction_of_outside_code_population_rejected"],2),pct(r["accepted_target_fidelity"],2),pct(r["unconditional_target_fidelity"],2)]for r in filtering],[1.45,2.25,1,1]),
        P("Here 'bad' means outside the code at delivery, assessed from joint density-matrix populations rather than assigning a hidden good/bad label to each shot. Postselection removes a substantial fraction of that population. BK also rejects 5/6 of ideal inputs, so its rejection cannot be interpreted as noise detection alone. [S6]"),
    ],[
        H("DD, conclusions, and reproducibility"),
        P("In the original April-snapshot circuits, adding 2, 4 or 8 native X pulses to eligible idles does not improve the simulated fidelity. The full calibrated pulse noise is retained. These X-only tests are distinct from the later hardware XY4 experiment. [S7]"),
        T(["Original protocol","No DD","XX","X4","X8"],[[name(r)]+[pct(r["variants"][k]["metrics"]["fidelity"],3)for k in ["no_dd","xx","x4","x8"]]for r in dd],[1.6,1,1,1,1]),
        P("The memoryless T1/T2 model contains no slowly varying phase noise for an echo to refocus. Added imperfect pulses can therefore worsen its result. This does not establish that DD is ineffective on hardware, nor that hardware DD removes T1 relaxation."),
        T(["Original circuit control","Full-noise F","All delay noise off F","Improvement (pp)"],[[name(r),pct(r["full"]["fidelity"],3),pct(r["no_delay"]["fidelity"],3),pp(r["no_delay"]["fidelity"]-r["full"]["fidelity"],3)]for r in original],[1.6,1.2,1.5,1.3]),
        P("Conclusion: syndrome-extraction idles are a major removable source of infidelity in this model, particularly the final unobserved data wait. Noisy extraction gates explain most remaining outside-code population when these idles are suppressed. Readout correction improves the quality estimate; it does not restore the delivered quantum state or retrospectively repair accept/reject decisions."),
        N("B1 and B2 finish at different times. Their reported fidelities do not establish a same-consumption-time error-detection advantage. The fixed physical patch was not optimized. Later scheduling alternatives were marginal and are not the basis of these conclusions."),
        H("Evidence and verification"),
        P("Archived checks: ideal circuits recover fidelity one; the idle channel matches exp(-t/T1) and exp(-t/T2) within 3.3×10^-16; explicit phase-noise replay matches the automatic-delay model within 10^-11; frozen QPY hashes and ideal-extraction projection controls pass. These reports reformat archived results; no simulation or hardware experiment was rerun."),
        P("Environment: Qiskit 2.4.1, Aer 0.17.2, qiskit-ibm-runtime 0.47.0, NumPy 1.26.4 in quantum-jax. Numeric inputs and SHA-256 hashes are indexed in [summary_report_sources.json](summary_report_sources.json)."),
        H("Source index"),
        P("[S1] [Original results](../results/noisy_aachen/noisy_results.json). [S2] [Qiskit timelines and timing](../results/noisy_aachen/timelines/timing_report.md). [S3] [Phase ablations](../results/noisy_aachen/syndrome_delay_ablation/results.json). [S4] [Residual gate controls](../results/noisy_aachen/syndrome_delay_ablation/residual_codespace_loss.json). [S5] [Readout and finite-shot estimates](../results/noisy_aachen/readout_and_fidelity.json). [S6] [Syndrome filtering](../results/noisy_aachen/syndrome_filtering.json). [S7] [DD tests](../results/noisy_aachen/dd_full_noise/results.json)."),
        P("Model documentation: [IBM FakeAachen](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/fake-provider-fake-aachen), [Aer NoiseModel](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.NoiseModel.html), [thermal relaxation](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.thermal_relaxation_error.html)."),
    ]]
    return pages


def four_qubit_hardware_pages():
    runs=mit["runs"]
    modes=[("raw_recorded_zero","Raw; recorded 000"),("data_corrected_recorded_zero","Data corrected; recorded 000"),("joint_corrected_true_zero","Joint corrected; latent true 000")]
    return [[
        H("Final measured performance"),
        P("We ran the four-qubit B2_ED encoded-H preparation on IBM Aachen twice: a baseline with no DD, followed by the same circuit with explicit XY4 pulses inserted into long idles. The primary final result is the fidelity estimate after correcting final data readout, while keeping the actual recorded-000 acceptance rule."),
        I(FIG/"hardware_fidelity.png","Figure 1. Measured accepted-state fidelity estimates. Error bars are 95% normal sampling intervals with calibration treated as fixed. Readout correction changes the estimate, not the prepared state. [H1]",190),
        T(["Primary metric","No DD","XY4"],[
            ["Data-readout-corrected fidelity"]+[pct(r["fidelity"]["data_corrected_recorded_zero"]["estimate"],3)for r in runs],
            ["95% sampling interval (%)"]+[ci(r["fidelity"]["data_corrected_recorded_zero"]["normal_95_interval"],3)for r in runs],
            ["Raw accepted fidelity score"]+[pct(r["fidelity"]["raw_recorded_zero"]["estimate"],3)for r in runs],
            ["Dedicated acceptance"]+[f"{a['primary_acceptance']['accepted_shots']}/2048 = {pct(a['primary_acceptance']['acceptance'],3)}"for a in analyses],
            ["Acceptance Wilson 95% interval (%)"]+[ci(a["primary_acceptance"]["acceptance_wilson_95_interval"],3)for a in analyses],
            ["Data-corrected codespace score"]+[pct(r["codespace_score"]["data_corrected_recorded_zero"]["estimate"],3)for r in runs],
        ],[2.6,1.5,1.5]),
        P("The observed XY4 minus baseline difference is "+pp(mit["comparison"]["data_corrected_recorded_zero"]["estimate"],3)+" percentage points after data correction, with 95% sampling interval "+ci(mit["comparison"]["data_corrected_recorded_zero"]["normal_95_interval"],3)+" points. The raw-score difference is +4.511 points. Dedicated acceptance changes by +2.539 points; its difference interval includes zero."),
        N("The two jobs were about 24.5 minutes apart, with no interleaving or randomization. Sampling intervals exclude drift and calibration error. The observed improvement supports XY4 as a promising intervention, but does not isolate its causal effect from time-dependent hardware changes."),
    ],[
        H("Experiment and execution details"),
        P("Target: |HL〉 = cos(π/8)|0L〉 + sin(π/8)|1L〉, with |0L〉 = (|0000〉+|1111〉)/√2 and |1L〉 = (|0011〉+|1100〉)/√2. The [[4,1,2]] stabilizers are XXXX, IIZZ and ZZII, recorded as syndrome bits 0, 1 and 2. The displayed bitstring is s2s1s0. Accept only 000; no active Pauli recovery is applied."),
        T(["Run","IBM job ID","Execution on 14 Oct 2026 (Pacific)"],[
            ["No DD",runs[0]["job_id"],"20:21:53-20:22:11"],
            ["XY4",runs[1]["job_id"],"20:46:23-20:46:42"],
        ],[.6,2,2.6]),
        P("Each job contained 24 circuits × 2,048 attempted shots = 49,152 shots: one dedicated acceptance-only circuit and 23 target-projector Pauli settings. Total: 98,304 attempted shots. A frozen preparation prefix is shared across settings; only the final analysis basis changes. Joint data and syndrome counts are archived, preserving their correlations. [H2, H3]"),
        T(["Circuit property","Value"],[
            ["Backend / calibration","ibm_aachen / live snapshot 14 Oct 2026, 20:06:27 Pacific"],
            ["Fixed local patch","[3, 2, 4, 16, 1, 5, 23, 0, 6]"],
            ["Output data q0, q1, q2, q3","Physical qubits [5, 3, 2, 16]"],
            ["Output syndrome s0, s1, s2","Physical qubits [1, 4, 23]"],
            ["No-DD preparation resources","39 CZ, 67 sx, 40 virtual rz; 31 delays"],
            ["Preparation through syndrome delivery","2.840 µs before readout + 2.180 µs readout = 5.020 µs"],
            ["Runtime execution options","Initialize qubits; 250 µs inter-shot rep_delay; classified readout"],
            ["Other automated suppression","Gate/measurement twirling OFF; automatic Runtime DD OFF"],
        ],[2,3.6]),
        P("The 5.020 µs delivery time excludes the subsequent destructive basis-analysis and data-measurement suffix. The 250 µs repetition delay is between shots, not a data-storage wait within the preparation. Each variant used one SamplerV2 job, with no Session or Batch."),
        N("Both archived manifests use the same calibration file. This helps match the comparison but does not show that the hardware noise was constant between jobs. Both circuit sets passed ideal statevector and ISA validation; all 24 ideal settings agreed to approximately 10^-15."),
    ],[
        H("Error suppression: postselection and XY4"),
        T(["Strategy actually used","Purpose","What it changes"],[
            ["Syndrome postselection","Discard detected faults","Retains recorded-000 runs; reduces yield"],
            ["Explicit XY4 during eligible idles","Refocus slow phase / coherent idle noise","Changes physical evolution before state delivery"],
            ["Offline data-readout correction","Remove assignment bias in fidelity estimates","Changes statistical weights; acceptance unchanged"],
        ],[1.55,1.8,2.25]),
        P("XY4 uses Xπ, Yπ, Xπ, Yπ in each eligible noninitial idle of at least 500 ns. Nine windows were protected, adding 36 native 32 ns X pulses and 36 virtual Rz operations, with no additional CZ gates. A Y pulse is synthesized by virtual Rz(-π/2), native X, then virtual Rz(+π/2). Initial and post-measurement idles are skipped."),
        I(FIG/"hardware_xy4_readout_timing.png","Figure 2. Actual saved XY4 pulse positions in the terminal readout window. Other protected windows occur earlier during extraction. This is an instruction-timing view, not an analog waveform. [H3]",215),
        P("All four output data qubits receive XY4 during the 2.180 µs ancilla-readout wait. Pulse starts are 3.096, 3.640, 4.188 and 4.732 µs from circuit start; each pulse is 0.032 µs. Free-gap fractions are approximately 1/8, 1/4, 1/4, 1/4, 1/8, rounded to the device clock."),
        P("The original gate and measurement timestamps, mapping, and total 5.020 µs preparation duration are preserved. The physical aim is to refocus slowly varying phase errors while the state would otherwise idle. DD does not reverse irreversible T1 decay, fix already faulty entangling gates, or correct classical readout assignments."),
        N("No adaptive syndrome recovery, repeated checks, flag-ancilla protocol, zero-noise extrapolation, or gate twirling was used. The flag-circuit paper discussed separately remains a possible future experiment, not an explanation of these results."),
        P("Background: [IBM error-suppression guidance](https://quantum.cloud.ibm.com/learning/en/courses/tools-for-quantum-advantage/error-suppression)."),
    ],[
        H("How fidelity is estimated"),
        P("We estimate overlap with the full four-qubit target, including loss outside the code subspace. This is not full density-matrix tomography and not an estimate from logical X, Y and Z alone."),
        P("Let P_code be the stabilizer projector, XL = IIXX and ZL = IZIZ. The target projector is O = P_code [I + (XL + ZL)/√2] / 2 = Σ c_P P. Its identity coefficient is 1/16, and 23 nonidentity Pauli coefficients are nonzero."),
        P("For each Pauli P, prepare a fresh state, read the syndrome, wait for its readout to finish, and measure the data in the required basis: H before Z-readout for X, S† then H for Y, and direct Z-readout for Z. Identity factors do not enter the parity. Compute the mean parity only among the recorded-000 shots in that setting. Then F_hat = 1/16 + Σ_(P ≠ I) c_P mean(parity_P | recorded 000)."),
        H("Offline assignment correction"),
        P("For each data qubit, the saved directional calibration supplies A[true,recorded] = [[1-e01,e01],[e10,1-e10]]. Solve A w = (1,-1), and replace raw parity by the product of the appropriate single-bit weights. Identity factors have weight one. This independent-qubit tensor model is implemented directly; no M3 package or new calibration experiment was used. [H1]"),
        T(["Output data bit","Physical qubit","P(record 1 | true 0)","P(record 0 | true 1)"],[[str(r["bit"]),str(r["physical_qubit"]),pct(r["p_record_1_given_0"],3),pct(r["p_record_0_given_1"],3)]for r in runs[0]["calibration"]["data"]],[1.2,1.1,1.65,1.65]),
        P("The primary correction uses only final data assignment matrices. It neither changes which shots passed nor retroactively fixes syndrome decisions. It also does not correct noisy basis gates, crosstalk, correlated readout, or the quantum-state preparation itself. Directional calibration entries were measured around 19:34 Pacific, roughly 48 and 72 minutes before the two jobs."),
        H("Uncertainty and the syndrome-corrected diagnostic"),
        P("Variance is propagated from per-setting parity or corrected-weight samples. Normal 95% intervals are cross-checked with 10,000 multinomial bootstrap resamples of the joint histograms. Within-job raw/corrected comparisons are paired; the two jobs are treated as independent. These quantify shot noise only, conditional on fixed calibration and independent shots."),
        P("A secondary analysis also inverts syndrome assignment and uses all recorded syndromes to estimate the latent true-000 conditional ensemble. Its corrected numerator/denominator covariance is included. That answers a different question from the quality of the actually retained shots; it cannot identify and repair individual wrongly accepted or rejected shots."),
        T(["Fidelity estimate","No DD","XY4"],[[label]+[pct(r["fidelity"][k]["estimate"],3)for r in runs]for k,label in modes],[3.2,1.1,1.1]),
    ],[
        H("What the results establish, and their limits"),
        P("Postselection separates a substantially higher-quality ensemble from the rejected outcomes. The same target-projector estimator gives the following raw scores. They still include final measurement bias. [H2, H3]"),
        T(["Raw target score","No DD","XY4"],[[label]+[pct(a["groups"][key]["raw_fidelity"],3)for a in analyses]for key,label in [("all","All outcomes"),("accepted","Recorded 000 accepted"),("rejected","Rejected outcomes")]],[2.6,1.3,1.3]),
        P("This demonstrates useful statistical filtering within each executed syndrome circuit. It does not demonstrate an end-to-end advantage over a separately run B1 circuit: there was no hardware B1 control at a matched consumption time. Nor do the data identify exactly how many physical fault events were corrected; no active recovery was performed."),
        H("Model replay versus hardware"),
        T(["Quantity","No DD","XY4"],[
            ["Saved-calibration model acceptance"]+[pct(r["acceptance"],3)for r in replays],
            ["Hardware dedicated acceptance"]+[pct(a["primary_acceptance"]["acceptance"],3)for a in analyses],
            ["Model raw measurement score"]+[pct(r["raw_measurement_score"],3)for r in replays],
            ["Hardware raw measurement score"]+[pct(a["groups"]["accepted"]["raw_fidelity"],3)for a in analyses],
            ["Model exact preparation fidelity"]+[pct(r["exact_model_preparation_fidelity"],3)for r in replays],
        ],[2.8,1.2,1.2]),
        P("The replay uses each frozen hardware circuit and its saved live calibration, rather than the April packaged snapshot. It predicts a small XY4 penalty, whereas hardware shows a higher score in the XY4 job. The gap is consistent with missing noise or control effects in the simple model, but these data cannot uniquely attribute it to slow phase noise, idles, crosstalk, drift, or another mechanism. [H4]"),
        P("No hardware density matrix was reconstructed. The data-corrected code-projector scores, 79.018% and 84.857%, are estimates with the same calibration and measurement assumptions. Their ratio with a fidelity estimate is not presented as an exact within-code state fidelity."),
        N("Final performance statement: XY4 achieved a data-readout-corrected accepted target fidelity estimate of 83.011%, with 95% shot-only interval [81.684%, 84.338%], at measured dedicated acceptance 50.977%. The baseline estimate was 77.659%. No claim of fault tolerance, isolated DD causality, or a complete hardware error budget follows from these two jobs."),
        H("Provenance and reproduction"),
        P("Verified from archived joint counts, raw Runtime results, manifests, calibration files, and analysis outputs. No new IBM jobs were submitted for this report. [H1] [Readout analysis](../results/hardware_b2ed/readout_mitigation_20261014/readout_mitigation.json). [H2] [Baseline analysis](../results/hardware_b2ed/four_pilot_20261014/analysis_extended.json). [H3] [XY4 manifest](../results/hardware_b2ed/four_xy4_20261014/manifest.json) and [analysis](../results/hardware_b2ed/four_xy4_20261014/analysis_extended.json). [H4] [Baseline replay](../results/hardware_b2ed/four_pilot_20261014/calibration_replay.json) and [XY4 replay](../results/hardware_b2ed/four_xy4_20261014/calibration_replay.json)."),
        P("Methods: [IBM readout mitigation](https://quantum.cloud.ibm.com/docs/en/tutorials/readout-error-mitigation-sampler). Code: [hardware estimator](../src/logical_magic/ibm_b2ed.py), [XY4 insertion](../src/logical_magic/ibm_dd.py), [offline correction](../scripts/mitigate_ibm_b2ed_readout.py). [Executed hardware notebook](../notebooks/03_ibm_aachen_hardware_results.ipynb). Input hashes: [source manifest](summary_report_sources.json)."),
    ]]


def five_results():
    result = read(HW / "five_comparison_20261014/analysis.json")
    for r in result["runs"]:
        directory = HW / Path(r["bundle"]).name
        for filename, digest in r["input_sha256"].items():
            assert hashlib.sha256((directory/filename).read_bytes()).hexdigest() == digest
            INPUTS[str((directory/filename).relative_to(ROOT))] = digest
    return result


def make_hardware_figures():
    """New overview only; preserve the archived simulation and four-qubit figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    result = five_results()
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), layout="constrained", sharey=True)
    for ax, title, runs in zip(axes, ["[[4,1,2]] logical H", "[[5,1,3]] logical tetrahedral"], [mit["runs"], result["runs"]]):
        for i, (mode, label, color) in enumerate([("raw_recorded_zero", "Raw", "#738da6"), ("data_corrected_recorded_zero", "Data corrected", "#208a88")]):
            x = np.arange(2)+(i-.5)*.30
            y = np.array([r["fidelity"][mode]["estimate"]*100 for r in runs])
            err = np.array([r["fidelity"][mode]["standard_error"]*100*1.959964 for r in runs])
            ax.bar(x, y, width=.28, color=color, label=label, yerr=err, capsize=3)
            for xx, yy, ee in zip(x, y, err):
                ax.text(xx, yy+ee+1, f"{yy:.2f}", ha="center", fontsize=9)
        ax.set(xticks=[0, 1], xticklabels=["No DD", "XY4"], title=title, ylim=(0, 100))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Accepted target fidelity estimate (%)")
    axes[1].legend(frameon=True, facecolor="white", framealpha=1, loc="lower right", fontsize=9)
    for ext in ("png", "svg"):
        fig.savefig(FIG/f"hardware_all_codes_fidelity.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def hardware_pages():
    five = five_results()
    runs = five["runs"]
    raw, corrected, joint = "raw_recorded_zero", "data_corrected_recorded_zero", "joint_corrected_true_zero"
    diff = five["comparison"]["fidelity"][corrected]
    accept_diff = five["comparison"]["dedicated_acceptance"]["recorded_zero"]
    all_rows = []
    for label, values, acceptance in [("Four / H", mit["runs"], analyses), ("Five / tetrahedral", runs, runs)]:
        for variant, r, a in zip(["No DD", "XY4"], values, acceptance):
            all_rows.append([label, variant, pct(r["fidelity"][raw]["estimate"]),
                             pct(r["fidelity"][corrected]["estimate"]), ci(r["fidelity"][corrected]["normal_95_interval"]),
                             pct(a["primary_acceptance"]["acceptance"])])
    pages = [[
        H("Four completed hardware experiments"),
        P("The four-qubit encoded H state and the five-qubit encoded positive tetrahedral state were each prepared with B2_ED on ibm_aachen, with and without explicit XY4. Both comparisons show higher accepted-state fidelity estimates in the XY4 job. The primary estimator corrects final data readout while retaining the actual recorded-zero syndrome selection."),
        I(FIG/"hardware_all_codes_fidelity.png", "Overview. Raw and data-readout-corrected estimates from the four completed jobs. Error bars are 95% sampling intervals with calibration fixed. Each code has its own target, preparation circuit and calibration snapshot.", 205),
        T(["Code / target", "DD", "Raw F", "Corrected F", "95% CI (%)", "Acceptance"], all_rows, [1.55, .6, .8, 1.0, 1.35, 1.0]),
        P("Five-qubit addition: corrected fidelity rises by "+pp(diff["estimate"])+" pp, with 95% sampling interval "+ci(diff["normal_95_interval"])+" pp. The earlier four-qubit difference was +5.35 pp [3.43, 7.28]. Dedicated acceptance differences include zero in both comparisons."),
        N("These are sequential job comparisons, with no interleaving or randomization. Shot-only intervals exclude calibration uncertainty and hardware drift. The four- and five-qubit results are not a controlled code-ranking experiment: targets, circuit sizes, checks and calibration snapshots differ. No active recovery was implemented."),
        P("This update analyzes existing results only: 360,448 attempted hardware shots across four jobs. No new IBM job or noise simulation was run for this report update. The five-qubit results follow; the earlier four-qubit analysis is retained in full."),
    ], [
        H("Five-qubit B2_ED: protocol and measured performance"),
        P("Target: the [[5,1,3]] encoded positive tetrahedral state, with logical Bloch vector (1,1,1)/sqrt(3). The four checks are XZZXI, IXZZX, XIXZZ and ZXIXZ in syndrome-bit order s0 through s3. Accept displayed s3s2s1s0 = 0000. This is encoding followed by detection and rejection; it does not exercise the code's single-error correction capability through active recovery."),
        T(["Quantity", "No DD", "XY4"], [
            ["Raw accepted target fidelity"]+[pct(r["fidelity"][raw]["estimate"],3) for r in runs],
            ["Data-corrected accepted fidelity"]+[pct(r["fidelity"][corrected]["estimate"],3) for r in runs],
            ["Corrected 95% sampling CI (%)"]+[ci(r["fidelity"][corrected]["normal_95_interval"],3) for r in runs],
            ["Dedicated acceptance"]+[f"{r['primary_acceptance']['accepted_shots']}/2048 = {pct(r['primary_acceptance']['acceptance'],3)}" for r in runs],
            ["Acceptance Wilson 95% CI (%)"]+[ci(r["primary_acceptance"]["acceptance_wilson_95_interval"],3) for r in runs],
            ["Pooled acceptance; supplementary"]+[pct(r["pooled_acceptance"]["estimate"],3) for r in runs],
        ], [2.6, 1.6, 1.6]),
        P("Each job used 63 nonidentity target-projector Pauli settings and one dedicated acceptance circuit, all at 2,048 shots: 131,072 attempted shots/job. Fidelity is F_hat = 1/32 + sum c_P mean(parity_P | recorded 0000), using the full five-qubit target projector. No additional codespace filtering is applied."),
        T(["Run", "IBM job ID", "Execution, 14 Oct 2026 Pacific"], [
            ["XY4 (first)", runs[1]["job_id"], "22:13:52-22:14:29"],
            ["No DD (second)", runs[0]["job_id"], "22:20:49-22:21:30"],
        ], [.9, 2.25, 2.45]),
        P("The submissions were 6.95 minutes apart. The original preparation and routing are shared: 106 CZ gates, 181 sx gates, 95 virtual rz gates and 2 X gates before DD. Final data bits map to [6,5,4,16,2]; syndrome bits map to [0,23,1,3]."),
        P("XY4 protects 28 eligible idle windows with 112 additional native X pulses and 112 virtual Rz gates. This includes all five output qubits during the 2.180 microsecond syndrome-readout wait. Original gate timestamps and the 9.992 microsecond preparation-through-delivery duration are preserved. Automatic Runtime DD and gate/measurement twirling are disabled; rep_delay is 250 microseconds."),
        N("The corrected fidelity difference is "+pp(diff["estimate"],3)+" pp; normal CI "+ci(diff["normal_95_interval"],3)+" and independent-bootstrap CI "+ci(diff["bootstrap_95_interval"],3)+" pp. Dedicated acceptance changes by "+pp(accept_diff["estimate"],3)+" pp, CI "+ci(accept_diff["normal_95_interval"],3)+" pp; this acceptance result is inconclusive."),
    ], [
        H("Five-qubit analysis: readout correction and diagnostics"),
        P("The primary correction inverts only final data assignment errors using per-qubit directional matrices from the saved calibration. It preserves every recorded-0000 acceptance decision. The two jobs have byte-identical calibration files (snapshot 21:32:08 Pacific); the directional readout entries themselves date to 19:34:04-05, about 2 hours 40-47 minutes before execution. This calibration age is not included in the sampling intervals."),
        T(["Data bit", "Physical qubit", "P(record 1 | true 0)", "P(record 0 | true 1)"], [[str(r["bit"]), str(r["physical_qubit"]), pct(r["p_record_1_given_0"],3), pct(r["p_record_0_given_1"],3)] for r in runs[0]["calibration"]["data"]], [1,1,1.8,1.8]),
        T(["Diagnostic score", "No DD", "XY4"], [
            ["All-outcome raw target score"]+[pct(r["groups"]["all"]["raw_fidelity"],3) for r in runs],
            ["Rejected raw target score"]+[pct(r["groups"]["rejected"]["raw_fidelity"],3) for r in runs],
            ["Raw code-projector score"]+[pct(r["codespace_score"][raw]["estimate"],3) for r in runs],
            ["Data-corrected code-projector score"]+[pct(r["codespace_score"][corrected]["estimate"],3) for r in runs],
            ["Joint-corrected fidelity; latent 0000"]+[pct(r["fidelity"][joint]["estimate"],3) for r in runs],
        ], [3.2, 1.2, 1.2]),
        P("Joint data-plus-syndrome inversion estimates the latent true-0000 ensemble, which differs from the delivered recorded-0000 ensemble. It is diagnostic only. Code-projector scores estimate the population inside the code subspace under the same readout assumptions; no hardware density matrix or additional physical projection was reconstructed."),
        P("All 128 Runtime circuit results reproduce the archived shot histograms. Raw fidelity and standard error match both retrieval and independent bit-count calculations. Corrected observable scores agree with distribution inversion within 1.3e-15. Tensor condition numbers are 1.196 (data) and 1.305 (joint). Negative quasiprobabilities and estimates are retained without clipping."),
        P("Uncertainty uses 10,000 per-setting multinomial bootstrap replicates. Estimators share resamples within a job, and jobs have independent random streams. Normal and bootstrap intervals agree. Acceptance homogeneity across 64 settings gives chi-square(63) = 77.39, p = 0.105 (no DD) and 59.47, p = 0.603 (XY4); this does not rule out hardware drift."),
        N("The fidelity increase is much larger than estimated sampling noise, but cannot be uniquely assigned to DD from one sequential pair. No matched hardware B1 control, fault-tolerant verification, active correction, fresh readout-calibration experiment or five-qubit live-calibration noise replay was run."),
        P("Five-qubit sources: [comparison and uncertainty](../results/hardware_b2ed/five_comparison_20261014/analysis.json), [readable analysis](../results/hardware_b2ed/five_comparison_20261014/report.md), [no-DD extended analysis](../results/hardware_b2ed/five_pilot_20261014/analysis_extended.json), [XY4 extended analysis](../results/hardware_b2ed/five_xy4_20261014/analysis_extended.json), [offline analyzer](../scripts/analyze_ibm_five_b2ed.py). The executed [hardware notebook](../notebooks/03_ibm_aachen_hardware_results.ipynb) displays both code comparisons."),
    ]]
    for page in four_qubit_hardware_pages():
        pages.append([(b[0], "Four-qubit study: "+b[1]) if b[0] == "h" else b for b in page])
    return pages


def write_markdown(title, subtitle, pages, path):
    lines=[f"# {title}","",subtitle,"","Prepared 14 October 2026 (Pacific) from archived results. Percentages are rounded; changes are percentage points (pp)."," "]
    for page in pages:
        for b in page:
            if b[0]=="h":lines += ["## "+b[1],""]
            elif b[0]=="p":lines += [b[1],""]
            elif b[0]=="note":lines += ["> "+b[1],""]
            elif b[0]=="table":
                lines += ["| "+" | ".join(str(x).replace("|","\\|")for x in b[1])+" |", "|"+"|".join("---"for _ in b[1])+"|"]
                lines += ["| "+" | ".join(str(x).replace("|","\\|")for x in row)+" |"for row in b[2]]
                lines += [""]
            elif b[0]=="image":
                rel=os.path.relpath(b[1],OUT)
                lines += [f"![{b[2]}]({rel})","",b[2],""]
    path.write_text("\n".join(lines))


def write_pdf(title, subtitle, pages, path):
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    fonts=Path("/Users/runzhemo/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/libreoffice-headless/libreoffice/LibreOfficeDev.app/Contents/Resources/fonts/truetype")
    for family,file in [("Report","DejaVuSans.ttf"),("Report-Bold","DejaVuSans-Bold.ttf")]:
        if family not in pdfmetrics.getRegisteredFontNames():pdfmetrics.registerFont(TTFont(family,str(fonts/file)))
    pdfmetrics.registerFontFamily("Report",normal="Report",bold="Report-Bold",italic="Report",boldItalic="Report-Bold")
    styles={
        "p":ParagraphStyle("p",fontName="Report",fontSize=9.1,leading=13.0,spaceAfter=8,textColor=colors.HexColor("#203247")),
        "h":ParagraphStyle("h",fontName="Report-Bold",fontSize=13.2,leading=17,spaceBefore=5,spaceAfter=9,textColor=colors.HexColor("#173953"),keepWithNext=True),
        "note":ParagraphStyle("note",fontName="Report",fontSize=8.2,leading=11.5,spaceAfter=8,backColor=colors.HexColor("#edf3f6"),borderPadding=7),
        "cap":ParagraphStyle("cap",fontName="Report",fontSize=7.6,leading=10,spaceAfter=8,textColor=colors.HexColor("#536477")),
        "cell":ParagraphStyle("cell",fontName="Report",fontSize=7.8,leading=10.5),
        "th":ParagraphStyle("th",fontName="Report-Bold",fontSize=7.7,leading=10.3,textColor=colors.white),
        "title":ParagraphStyle("title",fontName="Report-Bold",fontSize=23,leading=27,spaceAfter=8,textColor=colors.HexColor("#173953")),
    }
    def rich(text):
        text=html.escape(str(text))
        def link(m):
            dest=html.unescape(m.group(2))
            if not dest.startswith("https://"):
                dest=os.path.relpath((OUT/dest).resolve(),PDF)
            return '<link href="'+html.escape(dest,quote=True)+'" color="#1f7084">'+m.group(1)+'</link>'
        return re.sub(r"\[([^\]]+)\]\(([^)]+)\)",link,text)
    story=[]; width=A4[0]-84
    for index,page in enumerate(pages):
        if index:story.append(PageBreak())
        else:
            story.append(Paragraph(title,styles["title"]))
            story.append(Paragraph(subtitle+" | 14 October 2026 (Pacific)",styles["cap"]))
        for b in page:
            kind=b[0]
            if kind in ["p","h","note"]:story.append(Paragraph(rich(b[1]),styles[kind]))
            elif kind=="table":
                rows=[[Paragraph(rich(v),styles["th" if i==0 else "cell"])for v in row]for i,row in enumerate([b[1]]+b[2])]
                weights=b[3] or [1]*len(b[1]); widths=[width*w/sum(weights)for w in weights]
                table=Table(rows,colWidths=widths,repeatRows=1,hAlign="LEFT")
                table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#284b70")),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f0f4f7"),colors.white]),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),("LINEBELOW",(0,-1),(-1,-1),.5,colors.HexColor("#c5d1db"))]))
                story.extend([table,Spacer(1,10)])
            elif kind=="image":
                img=Image(str(b[1]));scale=min(width/img.imageWidth,b[3]/img.imageHeight)
                img.drawWidth=img.imageWidth*scale;img.drawHeight=img.imageHeight*scale
                story.extend([img,Spacer(1,4),Paragraph(rich(b[2]),styles["cap"])])
    def footer(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor("#c5d1db"));canvas.line(42,38,A4[0]-42,38)
        canvas.setFont("Report",7);canvas.setFillColor(colors.HexColor("#536477"))
        canvas.drawString(42,25,"Qompute | "+subtitle+" | Archived evidence")
        canvas.drawRightString(A4[0]-42,25,str(doc.page));canvas.restoreState()
    SimpleDocTemplate(str(path),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=35,bottomMargin=49,title=title,author="Qompute project",pageCompression=1).build(story,onFirstPage=footer,onLaterPages=footer)


def make_reports(hardware_only=False):
    OUT.mkdir(exist_ok=True);PDF.mkdir(parents=True,exist_ok=True)
    docs=[("hardware_experiment_summary","IBM Aachen hardware experiments","Four- and five-qubit B2_ED: no DD versus XY4",hardware_pages())]
    if not hardware_only:
        docs.insert(0,("noisy_simulation_summary","IBM Aachen noisy simulation","Noise sources, timing, and controlled ablations",simulation_pages()))
    for stem,title,subtitle,pages in docs:
        write_markdown(title,subtitle,pages,OUT/f"{stem}.md")
        write_pdf(title,subtitle,pages,PDF/f"{stem}.pdf")
    for row in original:
        source = SIM / "circuits" / f"{row['code']}_{row['strategy']}.qpy"
        INPUTS[str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest={"purpose":"Reports derived from archived results; no new experiment execution", "inputs_sha256":INPUTS,"figures":"Frozen original QPY schedules redrawn with Qiskit timeline_drawer; other plots use archived JSON values","report_builder":"scripts/build_summary_reports.py","report_builder_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/"summary_report_sources.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"Created {len(docs)} Markdown and {len(docs)} PDF reports.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures",action="store_true")
    parser.add_argument("--reports",action="store_true")
    parser.add_argument("--hardware-only",action="store_true",help="Update hardware outputs while preserving simulation reports/figures")
    args=parser.parse_args()
    if args.figures:
        if not args.hardware_only:make_figures()
        make_hardware_figures()
    if args.reports:make_reports(hardware_only=args.hardware_only)
    if not(args.figures or args.reports):parser.error("choose --figures and/or --reports")
