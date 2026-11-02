using System.Text.Json;
using OpenRevelare.Core;
using OpenRevelare.DeepWb.Onnx;

internal static class AiSyntheticVerification
{
    // These phase functions and clean scenes are independent of the Python training generator.
    public static int Run(string output)
    {
        Directory.CreateDirectory(output);
        using var estimator = new OnnxNewtonEstimator();
        var report = new List<object>();
        foreach (string shape in new[] { "clean", "linear", "ellipse", "warped", "saddle", "crossed", "mono" })
        {
            const int w = 320, h = 256;
            var clean = new ImageBuffer(w, h); var observed = new ImageBuffer(w, h);
            for (int y = 0; y < h; y++) for (int x = 0; x < w; x++)
            {
                double u = x / (double)w, v = y / (double)h;
                double phase = shape switch
                {
                    "ellipse" => 100 * Math.Sqrt(Math.Pow(u - .3, 2) + .6 * Math.Pow(v - .65, 2) + .02),
                    "warped" => 90 * u + 9 * Math.Sin(8 * v) + 6 * u * v,
                    "saddle" => 120 * ((u - .5) * (u - .5) - .6 * (v - .5) * (v - .5)) + 25 * u,
                    _ => 95 * u + 32 * v,
                };
                for (int c = 0; c < 3; c++)
                {
                    int i = (y * w + x) * 3 + c;
                    clean.Data[i] = (float)(.24 + .08 * Math.Sin(3 * u + c) + .055 * Math.Cos(5 * v + c)
                        + .012 * Math.Sin(x * 1.7 + y * 1.2) + (x > 170 && y > 100 ? .06 : 0));
                    double fringe = shape == "clean" ? 0 : .11 * Math.Sin(phase * (shape == "mono" ? 1 : 1 + .14 * c) + (shape == "mono" ? 0 : c * 1.7));
                    if (shape == "crossed") fringe += .045 * Math.Sin(40 * u - 80 * v + c);
                    observed.Data[i] = clean.Data[i] * (float)Math.Exp(fringe);
                }
            }
            var repaired = new ImageBuffer(w, h); Array.Copy(observed.Data, repaired.Data, observed.Data.Length);
            var field = estimator.Estimate(observed, .5);
            NewtonRings.Apply(repaired, new() { NewtonEnabled = true, NewtonAutomatic = true, NewtonStrength = 1,
                NewtonLuminance = 1, NewtonField = field });
            double before = 0, after = 0, change = 0;
            for (int i = 0; i < clean.Data.Length; i++)
            {
                before += Math.Pow(Math.Log(observed.Data[i] / clean.Data[i]), 2);
                after += Math.Pow(Math.Log(repaired.Data[i] / clean.Data[i]), 2);
                change += Math.Abs(Math.Log(repaired.Data[i] / observed.Data[i]));
            }
            report.Add(new { Shape = shape, BeforeLogMse = before / clean.Data.Length, AfterLogMse = after / clean.Data.Length,
                MeanAbsoluteLogChange = change / clean.Data.Length, field.Coverage });
        }
        File.WriteAllText(Path.Combine(output, "synthetic-report.json"), JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
        return 0;
    }
}
