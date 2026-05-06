"""Plot forward / gradient timings collected by the benchmark scripts.

Each benchmark writes a JSON file with the following shape:

    {
      "entries": [
        {"label": "<framework>", "forward_us": <float>, "grad_us": <float>},
        ...
      ]
    }

This script concatenates the entries (in input-file order) and produces a
side-by-side bar chart using ROOT. By default it reads the three result
files produced by ``run_all.sh`` in the current directory.
"""

import argparse
import json

import ROOT


DEFAULT_INPUTS = [
    "results_torch.json",
    "results_jax.json",
    "results_sofie_clad.json",
]


def load_entries(paths):
    entries = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        entries.extend(data["entries"])
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=DEFAULT_INPUTS,
        help="Result JSON files to plot, in display order.",
    )
    parser.add_argument(
        "--out",
        default="plot.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--title",
        default="Forward vs Gradient Time per Sample (CPU) - MLP with ~70k parameters",
    )
    args = parser.parse_args()

    entries = load_entries(args.inputs)
    labels = [e["label"] for e in entries]
    forward_times = [float(e["forward_us"]) for e in entries]
    grad_times = [float(e["grad_us"]) for e in entries]
    n = len(labels)

    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)

    canvas = ROOT.TCanvas("c_bench", "benchmark", 1000, 600)
    canvas.SetLeftMargin(0.10)
    canvas.SetRightMargin(0.04)
    canvas.SetBottomMargin(0.22)
    canvas.SetTopMargin(0.10)
    canvas.SetGridy()

    h_forward = ROOT.TH1F("h_forward", args.title, n, 0, n)
    h_grad = ROOT.TH1F("h_grad", "", n, 0, n)
    for i, lab in enumerate(labels, start=1):
        h_forward.GetXaxis().SetBinLabel(i, lab)
        h_grad.GetXaxis().SetBinLabel(i, lab)
        h_forward.SetBinContent(i, forward_times[i - 1])
        h_grad.SetBinContent(i, grad_times[i - 1])

    h_forward.SetFillColor(ROOT.TColor.GetColor("#4C72B0"))
    h_forward.SetLineColor(ROOT.kBlack)
    h_grad.SetFillColor(ROOT.TColor.GetColor("#55A868"))
    h_grad.SetLineColor(ROOT.kBlack)

    bar_w = 0.4
    h_forward.SetBarOffset(0.05)
    h_forward.SetBarWidth(bar_w)
    h_grad.SetBarOffset(0.05 + bar_w + 0.05)
    h_grad.SetBarWidth(bar_w)

    ymax = max(max(forward_times), max(grad_times))
    h_forward.SetMaximum(ymax * 1.18)
    h_forward.SetMinimum(0)

    h_forward.GetYaxis().SetTitle("Time per sample [#mus]")
    h_forward.GetYaxis().SetTitleOffset(1.1)
    h_forward.GetXaxis().SetLabelSize(0.035)
    h_forward.GetXaxis().LabelsOption("v")

    h_forward.Draw("BAR")
    h_grad.Draw("BAR SAME")

    leg = ROOT.TLegend(0.78, 0.80, 0.95, 0.89)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.AddEntry(h_forward, "Forward", "f")
    leg.AddEntry(h_grad, "Gradient", "f")
    leg.Draw()

    latex = ROOT.TLatex()
    latex.SetTextSize(0.022)
    latex.SetTextAlign(21)
    pad = 0.012 * ymax
    for i in range(n):
        x_left = i + 0.05 + bar_w / 2
        x_right = i + 0.05 + bar_w + 0.05 + bar_w / 2
        latex.DrawLatex(x_left, forward_times[i] + pad, f"{forward_times[i]:.1f}")
        latex.DrawLatex(x_right, grad_times[i] + pad, f"{grad_times[i]:.1f}")

    canvas.SaveAs(args.out)
    print(f"Plot written to {args.out}")


if __name__ == "__main__":
    main()
