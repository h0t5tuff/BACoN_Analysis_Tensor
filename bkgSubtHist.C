

void bkgSubtHist()
{
    TFile *file = TFile::Open("0-4files-05_19-9.root");

    TTree *tree = (TTree*)file->Get("tree");


    Float_t pulse_timing;
    tree->SetBranchAddress("pulse_timing", &pulse_timing);

    TH1F *hist = new TH1F("hist", "PE Timing Distribution ch9:05_19_2025(0-4files)", 80, 500, 7500);

    // Fill the histogram from tree entries
    Long64_t nEntries = tree->GetEntries();
    for (Long64_t i = 0; i < nEntries; i++) {
        tree->GetEntry(i);
        hist->Fill(pulse_timing);
    }

    // Draw the histogram
    TCanvas *canvas1 = new TCanvas("canvas1", "Histogram from Tree", 800, 600);
    gPad->SetLogy();
    hist->Draw();

    double subt = 100.0;

    TH1F *hist_modified = (TH1F*)hist->Clone("hist_modified");
    hist_modified->SetTitle("Pulse Timing with bkgSubt");

    // Loop over all bins and subtract the constant value
    for (int i = 1; i <= hist_modified->GetNbinsX(); i++) {
        double bin_content = hist_modified->GetBinContent(i);
        double new_value = bin_content - subt;
        if (new_value < 0) new_value = 0;  // Ensure non-negative counts
        hist_modified->SetBinContent(i, new_value);
    }

    // Draw the modified histogram
    TCanvas *canvas2 = new TCanvas("canvas2", "Pulse Timing with bkgSubt", 800, 600);
    gPad->SetLogy();
    hist_modified->Draw();



}