using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenRevelare.Core;

namespace OpenRevelare.DeepWb.Onnx;

/// <summary>Optional local residual network. No Python, training or network access at app runtime.</summary>
public sealed class OnnxNewtonEstimator : INewtonEstimator
{
    public const string Contract = "openrevelare.newton.log-residual.v1";
    public static string DefaultModelPath => Environment.GetEnvironmentVariable("OPENREVELARE_NEWTON_MODEL")
        ?? Path.Combine(AppContext.BaseDirectory, "models", "newton_fringe_v1.onnx");
    private const int Tile = 192, Stride = 96;
    private readonly InferenceSession _session;
    private readonly string _inputName, _outputName;

    public OnnxNewtonEstimator(string? path = null)
    {
        path ??= DefaultModelPath;
        if (!File.Exists(path)) throw new FileNotFoundException("未找到 AI 去牛顿环模型。请按 docs/newton-ai.md 导出模型，或设置 OPENREVELARE_NEWTON_MODEL。", path);
        using var options = new SessionOptions { IntraOpNumThreads = Math.Clamp(Environment.ProcessorCount / 2, 1, 4), InterOpNumThreads = 1 };
        _session = new InferenceSession(path, options);
        try
        {
            var meta = _session.ModelMetadata.CustomMetadataMap;
            if (!meta.TryGetValue("contract", out var contract) || contract != Contract
                || !meta.TryGetValue("normalization", out var norm) || norm != "per-tile-log-mean-div4-eps1e-5"
                || _session.InputMetadata.Count != 1 || _session.OutputMetadata.Count != 1)
                throw new InvalidDataException("Unsupported AI Newton model contract.");
            _inputName = _session.InputMetadata.Keys.Single(); _outputName = _session.OutputMetadata.Keys.Single();
            if (!_session.InputMetadata[_inputName].Dimensions.SequenceEqual(new[] { 1, 3, Tile, Tile })
                || !_session.OutputMetadata[_outputName].Dimensions.SequenceEqual(new[] { 1, 4, Tile, Tile }))
                throw new InvalidDataException("Unsupported AI Newton tensor dimensions.");
        }
        catch { _session.Dispose(); throw; }
    }

    public NewtonField Estimate(ImageBuffer source, double sensitivity, CancellationToken cancellation = default)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (!double.IsFinite(sensitivity) || source.Width < 1 || source.Height < 1) throw new ArgumentException("Invalid AI source or sensitivity.");
        var image = Resample.Box(source, 768);
        int w = image.Width, h = image.Height, pixels = Tile * Tile;
        var residual = new ImageBuffer(w, h);
        var confidence = new float[w * h]; var weights = new float[w * h];
        var logs = new float[image.Data.Length];
        for (int i = 0; i < logs.Length; i++)
        {
            if (!float.IsFinite(image.Data[i])) throw new InvalidDataException("AI input contains non-finite pixels.");
            logs[i] = MathF.Log(Math.Max(1e-5f, image.Data[i]));
        }
        var window = Enumerable.Range(0, Tile).Select(i => (float)Math.Pow(Math.Sin(Math.PI * (i + .5) / Tile), 2)).ToArray();
        for (int oy = -Stride; oy < h; oy += Stride) for (int ox = -Stride; ox < w; ox += Stride)
        {
            cancellation.ThrowIfCancellationRequested();
            var input = new DenseTensor<float>(new[] { 1, 3, Tile, Tile });
            for (int c = 0; c < 3; c++)
            {
                double mean = 0;
                for (int y = 0; y < Tile; y++) for (int x = 0; x < Tile; x++)
                    mean += logs[(Reflect(oy + y, h) * w + Reflect(ox + x, w)) * 3 + c];
                mean /= pixels;
                for (int y = 0; y < Tile; y++) for (int x = 0; x < Tile; x++)
                    input[0, c, y, x] = (float)((logs[(Reflect(oy + y, h) * w + Reflect(ox + x, w)) * 3 + c] - mean) / 4);
            }
            using var result = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(_inputName, input) });
            var output = result.Single(v => v.Name == _outputName).AsTensor<float>();
            if (!output.Dimensions.SequenceEqual(new[] { 1, 4, Tile, Tile })) throw new InvalidDataException("AI returned invalid tensor dimensions.");
            var bias = new double[3];
            for (int c = 0; c < 3; c++)
            {
                for (int y = 0; y < Tile; y++) for (int x = 0; x < Tile; x++) bias[c] += output[0, c, y, x];
                bias[c] /= pixels;
            }
            for (int y = 0; y < Tile; y++) for (int x = 0; x < Tile; x++)
            {
                int px = ox + x, py = oy + y;
                if (px < 0 || py < 0 || px >= w || py >= h) continue;
                int at = py * w + px; float weight = window[x] * window[y];
                weights[at] += weight;
                for (int c = 0; c < 3; c++) residual.Data[at * 3 + c] += (float)(output[0, c, y, x] - bias[c]) * weight;
                confidence[at] += output[0, 3, y, x] * weight;
            }
        }
        cancellation.ThrowIfCancellationRequested();
        for (int i = 0; i < w * h; i++)
        {
            float norm = Math.Max(weights[i], 1e-8f), amplitude = 0;
            for (int c = 0; c < 3; c++)
            {
                residual.Data[i * 3 + c] /= norm;
                amplitude = Math.Max(amplitude, Math.Abs(residual.Data[i * 3 + c]));
            }
            // A broad mask probability alone must not modify a clean image. Demand a
            // meaningful predicted residual as well, especially at conservative sensitivity.
            confidence[i] = confidence[i] / norm * Math.Clamp(amplitude / (float)(.012 + .025 * (1 - Math.Clamp(sensitivity, 0, 1))), 0, 1);
        }
        return NewtonField.FromLogResidual(residual, confidence);
    }
    private static int Reflect(int p, int n) { int period = n * 2; p = ((p % period) + period) % period; return p < n ? p : period - 1 - p; }
    public void Dispose() => _session.Dispose();
}
