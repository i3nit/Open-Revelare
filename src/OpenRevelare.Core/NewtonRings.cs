using System.IO.Compression;
using System.Numerics;
using System.Text.Json.Serialization;

namespace OpenRevelare.Core;

/// <summary>Ordered source-space brush samples; overlapping samples form arbitrary painted shapes.</summary>
public readonly record struct NewtonStroke(double X, double Y, double Radius, bool Erase = false);

/// <summary>Immutable, bounded source-grid correction. Reused for preview, export and undo rather
/// than re-estimating fringes at each resolution. Channels are log RGB chroma, log luminance, confidence.</summary>
public sealed class NewtonField
{
    public int Width { get; }
    public int Height { get; }
    public (double X, double Y, double W, double H) SourceRect { get; }
    public string Id { get; } = Guid.NewGuid().ToString("N");
    private readonly float[] _data;
    [JsonIgnore] public double Coverage => Enumerable.Range(0, Width * Height).Count(i => _data[i * 5 + 4] > .185) / (double)(Width * Height);
    internal NewtonField(int width, int height, float[] data, (double X, double Y, double W, double H)? sourceRect = null)
    {
        if (width < 1 || height < 1 || data.Length != checked(width * height * 5))
            throw new ArgumentException("Invalid Newton field dimensions.");
        var box = sourceRect ?? (0d, 0d, 1d, 1d);
        if (!double.IsFinite(box.Item1 + box.Item2 + box.Item3 + box.Item4) || box.Item1 < 0 || box.Item2 < 0
            || box.Item3 <= 0 || box.Item4 <= 0 || box.Item1 + box.Item3 > 1.0000000001 || box.Item2 + box.Item4 > 1.0000000001)
            throw new ArgumentException("Invalid Newton source rectangle.");
        SourceRect = box;
        Width = width; Height = height; _data = new float[data.Length];
        // Quantise once, before the first preview, so saving/reopening reproduces exactly the
        // same correction rather than introducing a fresh rounding step on every round trip.
        for (int i = 0; i < data.Length; i++) _data[i] = (float)(Math.Round(Math.Clamp(data[i], -1, 1) * 16384) / 16384);
    }
    /// <summary>Convert a learned log-RGB interference estimate into the same immutable field
    /// used by the spectral path. A model never receives permission to replace image pixels.</summary>
    public static NewtonField FromLogResidual(ImageBuffer residual, IReadOnlyList<float> confidence)
    {
        ArgumentNullException.ThrowIfNull(residual);
        ArgumentNullException.ThrowIfNull(confidence);
        if (residual.Width < 1 || residual.Height < 1 || residual.Width > 1024 || residual.Height > 1024
            || confidence.Count != residual.PixelCount) throw new ArgumentException("Invalid AI field dimensions.");
        var values = new float[checked(residual.PixelCount * 5)];
        for (int i = 0; i < residual.PixelCount; i++)
        {
            float r = residual.Data[i * 3], g = residual.Data[i * 3 + 1], b = residual.Data[i * 3 + 2];
            float mask = confidence[i];
            if (!float.IsFinite(r + g + b + mask)) throw new InvalidDataException("AI returned non-finite values.");
            float luma = (r + g + b) / 3;
            values[i * 5] = Math.Clamp(r - luma, -.4f, .4f);
            values[i * 5 + 1] = Math.Clamp(g - luma, -.4f, .4f);
            values[i * 5 + 2] = Math.Clamp(b - luma, -.4f, .4f);
            values[i * 5 + 3] = Math.Clamp(luma, -.4f, .4f);
            values[i * 5 + 4] = Math.Clamp(mask, 0, 1);
        }
        return new NewtonField(residual.Width, residual.Height, values);
    }

    public NewtonField InSourceRect((double X, double Y, double W, double H) region)
        => new(Width, Height, _data, region);

    public float Sample(double x, double y, int channel)
    {
        if (x < SourceRect.X || y < SourceRect.Y || x > SourceRect.X + SourceRect.W || y > SourceRect.Y + SourceRect.H) return 0;
        x = (x - SourceRect.X) / SourceRect.W; y = (y - SourceRect.Y) / SourceRect.H;
        double fx = Math.Clamp(x * Width - .5, 0, Width - 1), fy = Math.Clamp(y * Height - .5, 0, Height - 1);
        int x0 = (int)fx, y0 = (int)fy, x1 = Math.Min(x0 + 1, Width - 1), y1 = Math.Min(y0 + 1, Height - 1);
        double dx = fx - x0, dy = fy - y0;
        return (float)((1 - dy) * ((1 - dx) * _data[(y0 * Width + x0) * 5 + channel] + dx * _data[(y0 * Width + x1) * 5 + channel])
            + dy * ((1 - dx) * _data[(y1 * Width + x0) * 5 + channel] + dx * _data[(y1 * Width + x1) * 5 + channel]));
    }
    private string? _encoded;
    public string Encode() => _encoded ??= EncodeCore();
    private string EncodeCore()
    {
        using var stream = new MemoryStream();
        using (var zip = new DeflateStream(stream, CompressionLevel.Optimal, true))
        using (var writer = new BinaryWriter(zip))
        {
            bool regional = SourceRect != (0d, 0d, 1d, 1d);
            writer.Write(regional ? 2 : 1); writer.Write(Width); writer.Write(Height);
            if (regional) { writer.Write(SourceRect.X); writer.Write(SourceRect.Y); writer.Write(SourceRect.W); writer.Write(SourceRect.H); }
            foreach (float v in _data) writer.Write((short)Math.Round(Math.Clamp(v, -1, 1) * 16384));
        }
        return Convert.ToBase64String(stream.ToArray());
    }
    public static NewtonField? Decode(string? encoded)
    {
        if (string.IsNullOrEmpty(encoded)) return null;
        if (encoded.Length > 16_000_000) throw new InvalidDataException("Newton correction is too large.");
        using var zip = new DeflateStream(new MemoryStream(Convert.FromBase64String(encoded)), CompressionMode.Decompress);
        using var reader = new BinaryReader(zip);
        int version = reader.ReadInt32();
        if (version != 1 && version != 2) throw new InvalidDataException("Unknown Newton correction version.");
        int width = reader.ReadInt32(), height = reader.ReadInt32();
        if (width < 1 || height < 1 || width > 1024 || height > 1024) throw new InvalidDataException("Invalid Newton correction size.");
        (double, double, double, double)? region = version == 2
            ? (reader.ReadDouble(), reader.ReadDouble(), reader.ReadDouble(), reader.ReadDouble()) : null;
        var values = new float[checked(width * height * 5)];
        for (int i = 0; i < values.Length; i++)
        {
            short value = reader.ReadInt16();
            if (value < -16384 || value > 16384 || (i % 5 == 4 && value < 0)) throw new InvalidDataException("Invalid Newton field value.");
            values[i] = value / 16384f;
        }
        if (zip.ReadByte() != -1) throw new InvalidDataException("Unexpected Newton correction payload.");
        return new NewtonField(width, height, values, region);
    }
}

/// <summary>Local, overlapping spectral defringing, with no centre, radius or circularity model.
/// Periodic subject detail can be indistinguishable from interference in a single exposure;
/// automatic masks are candidates and must remain editable.</summary>
public static class NewtonRings
{
    public static NewtonField Analyse(ImageBuffer source, double sensitivity = .5, CancellationToken cancellation = default, bool manual = false)
        => AnalyseCore(source, sensitivity, cancellation, manual, null);

    /// <summary>Analyse a separately decoded picture region: whole-scan preview boxing erases
    /// fine fringes before a detector can see them. Direction is tangent to the bands in source pixels.</summary>
    public static NewtonField AnalyseBands(ImageBuffer regionPixels, (double X, double Y, double W, double H) sourceRect,
        double directionRadians = 0, double sensitivity = .5, CancellationToken cancellation = default)
    {
        ArgumentNullException.ThrowIfNull(regionPixels);
        if (regionPixels.Width < 1 || regionPixels.Height < 1) throw new ArgumentException("Empty Newton source image.");
        if (!double.IsFinite(directionRadians)) throw new ArgumentException("Invalid band direction.");
        int factor = Resample.BoxFactor(regionPixels.Width, regionPixels.Height, 1024);
        // Integer boxing discards incomplete right/bottom cells. Stretching those lost pixels
        // back into the field shifts the phase of fine bands, creating a second fringe.
        var realised = (sourceRect.X, sourceRect.Y,
            sourceRect.W * (regionPixels.Width / factor * factor) / regionPixels.Width,
            sourceRect.H * (regionPixels.Height / factor * factor) / regionPixels.Height);
        return AnalyseCore(regionPixels, sensitivity, cancellation, true, directionRadians).InSourceRect(realised);
    }

    private static NewtonField AnalyseCore(ImageBuffer source, double sensitivity, CancellationToken cancellation, bool manual, double? bandDirection)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (source.Width < 1 || source.Height < 1) throw new ArgumentException("Empty Newton source image.");
        if (!double.IsFinite(sensitivity)) throw new ArgumentException("Invalid sensitivity.");
        sensitivity = Math.Clamp(sensitivity, 0, 1);
        bool bandsOnly = bandDirection is not null;
        var image = Resample.Box(source, bandsOnly ? 1024 : 768);
        double tangentX = Math.Cos(bandDirection ?? 0), tangentY = Math.Sin(bandDirection ?? 0);
        bool InDirection(int fx, int fy) => !bandsOnly || Math.Abs(fx * tangentX + fy * tangentY) <= Math.Max(1.25, Math.Abs(-fx * tangentY + fy * tangentX) * .32);
        int w = image.Width, h = image.Height;
        if (w < 8 || h < 8) return new NewtonField(Math.Max(1, w), Math.Max(1, h), new float[Math.Max(1, w * h) * 5]);
        var log = new float[w * h * 3];
        for (int i = 0; i < log.Length; i++)
            log[i] = (float)Math.Log(Math.Max(1e-6, float.IsFinite(image.Data[i]) ? image.Data[i] : 1e-6));
        var output = new float[w * h * 5];

        // Separate octave ranges avoid subtracting the same fringe twice. A large tile only
        // contributes long-period bands; small tiles follow rapidly changing fringe direction.
        foreach (int size in new[] { 64, 128, 256, 512 }.Where(n => n <= Math.Max(64, Math.Min(w, h) * 2)))
        {
            var layer = new float[w * h * 5];
            var weights = new float[w * h];
            double minimumFrequency = size == 64 ? 3.2 : 2.5;
            double maximumFrequency = size == 64 ? (bandsOnly ? 28 : 14) : size == 128 ? 6 : 5;
            int stride = size / 2;
            var window = Enumerable.Range(0, size).Select(i => Math.Sin(Math.PI * (i + .5) / size)).ToArray();
            for (int oy = -stride; oy < h; oy += stride) for (int ox = -stride; ox < w; ox += stride)
            {
                cancellation.ThrowIfCancellationRequested();
                var spectra = new Complex[3][];
                for (int c = 0; c < 3; c++)
                {
                    var values = new Complex[size * size];
                    double mean = 0;
                    for (int y = 0; y < size; y++) for (int x = 0; x < size; x++)
                        mean += log[(Reflect(oy + y, h) * w + Reflect(ox + x, w)) * 3 + c];
                    mean /= size * size;
                    for (int y = 0; y < size; y++) for (int x = 0; x < size; x++)
                        values[y * size + x] = (log[(Reflect(oy + y, h) * w + Reflect(ox + x, w)) * 3 + c] - mean) * window[x] * window[y];
                    Fft2(values, size, inverse: false); spectra[c] = values;
                }
                var power = new double[size * size];
                var bands = Enumerable.Range(0, size / 2).Select(_ => new List<double>()).ToArray();
                double total = 0;
                for (int y = 0; y < size; y++) for (int x = 0; x < size; x++)
                {
                    int fx = x <= size / 2 ? x : x - size, fy = y <= size / 2 ? y : y - size;
                    double radius = Math.Sqrt(fx * fx + fy * fy);
                    if (radius < minimumFrequency || radius > maximumFrequency) continue;
                    int p = y * size + x;
                    Complex luma = (spectra[0][p] + spectra[1][p] + spectra[2][p]) / 3;
                    double chroma = Complex.Abs(spectra[0][p] - spectra[1][p]) + Complex.Abs(spectra[2][p] - spectra[1][p]);
                    power[p] = chroma * chroma + .18 * luma.Magnitude * luma.Magnitude;
                    bands[(int)radius].Add(power[p]); if (InDirection(fx, fy)) total += power[p];
                }
                var baseline = bands.Select(a => { a.Sort(); return a.Count == 0 ? 0 : a[a.Count / 2]; }).ToArray();
                double SpectralPower(int fx, int fy)
                {
                    int at = ((fy + size) % size) * size + (fx + size) % size;
                    Complex mean = (spectra[0][at] + spectra[1][at] + spectra[2][at]) / 3;
                    double colour = Complex.Abs(spectra[0][at] - spectra[1][at]) + Complex.Abs(spectra[2][at] - spectra[1][at]);
                    return colour * colour + .18 * mean.Magnitude * mean.Magnitude;
                }
                var peaks = new List<(int X, int Y, double Power)>();
                foreach (int p in Enumerable.Range(0, power.Length).OrderByDescending(i => power[i]))
                {
                    if (power[p] <= 0 || peaks.Count == 6) break;
                    int x = p % size, y = p / size, fx = x <= size / 2 ? x : x - size, fy = y <= size / 2 ? y : y - size;
                    if (fy < 0 || (fy == 0 && fx < 0) || !InDirection(fx, fy)) continue;
                    int r = (int)Math.Sqrt(fx * fx + fy * fy);
                    double amplitude = Math.Sqrt(power[p]) / (size * size * .4);
                    if (amplitude < (bandsOnly ? .0015 : .008) * (1.8 - sensitivity) || power[p] < baseline[r] * (8 - 5 * sensitivity)
                        || power[p] / Math.Max(total, 1e-12) < .018 * (1.5 - sensitivity)) continue;
                    double frequency = Math.Sqrt(fx * fx + fy * fy);
                    int lowerX = (int)Math.Round(fx * (frequency - 2.5) / frequency), lowerY = (int)Math.Round(fy * (frequency - 2.5) / frequency);
                    int upperX = (int)Math.Round(fx * (frequency + 2.5) / frequency), upperY = (int)Math.Round(fy * (frequency + 2.5) / frequency);
                    // An isolated scene edge has a broad, decaying spectrum. A fringe needs
                    // a local maximum in FREQUENCY as well as the usual directional prominence.
                    if ((!manual || bandsOnly) && power[p] < (SpectralPower(lowerX, lowerY) + SpectralPower(upperX, upperY)) * (1.2 - .4 * sensitivity)) continue;
                    if (!manual)
                    {
                        Complex rg = spectra[0][p] - spectra[1][p], bg = spectra[2][p] - spectra[1][p];
                        bool monochrome = rg.Magnitude + bg.Magnitude < Complex.Abs((spectra[0][p] + spectra[1][p] + spectra[2][p]) / 3) * .2;
                        bool useRed = rg.Magnitude >= bg.Magnitude;
                        var profile = new double[size];
                        for (int i = 0; i < size; i++)
                        {
                            double t = (i - size / 2.0) * .8;
                            int px = Reflect((int)Math.Round(ox + size / 2.0 + t * fx / frequency), w);
                            int py = Reflect((int)Math.Round(oy + size / 2.0 + t * fy / frequency), h);
                            int at = (py * w + px) * 3;
                            profile[i] = monochrome ? (log[at] + log[at + 1] + log[at + 2]) / 3
                                : log[at + (useRed ? 0 : 2)] - log[at + 1];
                        }
                        double average = profile.Average(), threshold = Math.Max(.002, amplitude * .12);
                        int changes = 0, previous = 0;
                        for (int i = 1; i < size - 1; i++)
                        {
                            double value = (profile[i - 1] + profile[i] + profile[i + 1]) / 3 - average;
                            int sign = value > threshold ? 1 : value < -threshold ? -1 : 0;
                            if (sign == 0) continue;
                            if (previous != 0 && sign != previous) changes++;
                            previous = sign;
                        }
                        // Reflection/window sidelobes from one step edge can look like a narrow
                        // FFT peak. A real repeated fringe also alternates in the spatial domain.
                        if (changes < 4) continue;
                    }
                    if (peaks.Any(a => Math.Pow(a.X - fx, 2) + Math.Pow(a.Y - fy, 2) < 9)) continue;
                    peaks.Add((fx, fy, power[p]));
                }
                var selected = new float[size * size];
                foreach (var peak in peaks) for (int y = 0; y < size; y++) for (int x = 0; x < size; x++)
                {
                    int fx = x <= size / 2 ? x : x - size, fy = y <= size / 2 ? y : y - size;
                    double r = Math.Sqrt(fx * fx + fy * fy);
                    if (r < minimumFrequency - .5 || r > maximumFrequency + .5 || !InDirection(fx, fy)) continue;
                    double d2 = Math.Min(Math.Pow(fx - peak.X, 2) + Math.Pow(fy - peak.Y, 2),
                                         Math.Pow(fx + peak.X, 2) + Math.Pow(fy + peak.Y, 2));
                    selected[y * size + x] = Math.Max(selected[y * size + x], (float)Math.Exp(-d2 / 2.0));
                }
                double phaseSupport = 0;
                foreach (var peak in peaks)
                {
                    int at = ((peak.Y + size) % size) * size + (peak.X + size) % size;
                    Complex rg = spectra[0][at] - spectra[1][at], bg = spectra[2][at] - spectra[1][at];
                    double phase = Math.Abs((rg * Complex.Conjugate(bg)).Imaginary) / Math.Max(rg.Magnitude * bg.Magnitude, 1e-12);
                    phaseSupport += peak.Power * (.25 + .75 * phase);
                }
                var luminance = new Complex[size * size];
                for (int p = 0; p < selected.Length; p++)
                {
                    Complex common = (spectra[0][p] + spectra[1][p] + spectra[2][p]) / 3;
                    luminance[p] = common * selected[p];
                    for (int c = 0; c < 3; c++) spectra[c][p] = (spectra[c][p] - common) * selected[p];
                }
                if (peaks.Count > 0)
                {
                    foreach (var spectrum in spectra) Fft2(spectrum, size, inverse: true);
                    Fft2(luminance, size, inverse: true);
                }
                double concentration = peaks.Sum(p => p.Power) / Math.Max(total, 1e-12);
                float confidence = (float)Math.Clamp((concentration - .045) / .16, 0, 1);
                // Opponent-colour phase delay raises confidence, but is not mandatory:
                // achromatic and nearly in-phase Newton fringes also occur.
                if (!manual && peaks.Count > 0)
                    confidence *= (float)(phaseSupport / Math.Max(peaks.Sum(p => p.Power), 1e-12));
                for (int y = 0; y < size; y++) for (int x = 0; x < size; x++)
                {
                    int dx = ox + x, dy = oy + y;
                    if (dx < 0 || dy < 0 || dx >= w || dy >= h) continue;
                    int p = dy * w + dx, q = y * size + x;
                    double win = window[x] * window[y], weight = win * win;
                    weights[p] += (float)weight;
                    if (peaks.Count == 0) continue;
                    for (int c = 0; c < 3; c++) layer[p * 5 + c] += (float)(spectra[c][q].Real * win);
                    layer[p * 5 + 3] += (float)(luminance[q].Real * win);
                    layer[p * 5 + 4] += confidence * (float)weight;
                }
            }
            for (int p = 0; p < w * h; p++) for (int c = 0; c < 5; c++)
            {
                float value = layer[p * 5 + c] / Math.Max(weights[p], 1e-6f);
                output[p * 5 + c] = c == 4 ? Math.Clamp(Math.Max(output[p * 5 + c], value), 0, 1)
                    : Math.Clamp(output[p * 5 + c] + value, -.4f, .4f);
            }
        }
        return new NewtonField(w, h, output);
    }

    public static float[] Mask(int width, int height, NewtonField? field, bool automatic, IReadOnlyList<NewtonStroke> strokes,
        (double X, double Y, double W, double H)? region = null)
    {
        var box = region ?? (0d, 0d, 1d, 1d);
        var mask = new float[width * height];
        if (automatic && field is not null) for (int y = 0; y < height; y++) for (int x = 0; x < width; x++)
            mask[y * width + x] = Math.Clamp((field.Sample(box.Item1 + (x + .5) / width * box.Item3,
                box.Item2 + (y + .5) / height * box.Item4, 4) - .15f) / .35f, 0, 1);
        foreach (var s in strokes)
        {
            if (!double.IsFinite(s.X + s.Y + s.Radius) || s.Radius <= 0 || s.Radius > .5) continue;
            // Distances use the full source's short axis, not the current slice's aspect.
            double fw = width / box.Item3, fh = height / box.Item4, radius = s.Radius * Math.Min(fw, fh);
            double cx = (s.X - box.Item1) * fw - .5, cy = (s.Y - box.Item2) * fh - .5;
            int x0 = Math.Max(0, (int)Math.Floor(cx - radius)), x1 = Math.Min(width - 1, (int)Math.Ceiling(cx + radius));
            int y0 = Math.Max(0, (int)Math.Floor(cy - radius)), y1 = Math.Min(height - 1, (int)Math.Ceiling(cy + radius));
            for (int y = y0; y <= y1; y++) for (int x = x0; x <= x1; x++)
            {
                float alpha = (float)Math.Clamp((1 - Math.Sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy)) / radius) / .25, 0, 1);
                int p = y * width + x;
                mask[p] = s.Erase ? mask[p] * (1 - alpha) : Math.Max(mask[p], alpha);
            }
        }
        return mask;
    }

    public static void Apply(ImageBuffer image, FrameParams p, (double X, double Y, double W, double H)? region = null)
    {
        if (!p.NewtonEnabled || p.NewtonField is not { } field || !double.IsFinite(p.NewtonStrength + p.NewtonLuminance) || p.NewtonStrength <= 0) return;
        var box = region ?? p.NewtonSourceRect ?? (0d, 0d, 1d, 1d);
        var mask = Mask(image.Width, image.Height, field, p.NewtonAutomatic, p.NewtonStrokes, box);
        Parallel.For(0, image.Height, y =>
        {
            for (int x = 0; x < image.Width; x++)
            {
                int at = y * image.Width + x;
                double amount = mask[at] * Math.Clamp(p.NewtonStrength, 0, 1);
                if (amount <= 0) continue;
                double u = box.Item1 + (x + .5) / image.Width * box.Item3, v = box.Item2 + (y + .5) / image.Height * box.Item4;
                double luma = field.Sample(u, v, 3) * Math.Clamp(p.NewtonLuminance, 0, 1);
                for (int c = 0; c < 3; c++) image.Data[at * 3 + c] *= (float)Math.Exp(-Math.Clamp(field.Sample(u, v, c) + luma, -.5, .5) * amount);
            }
        });
    }
    private static int Reflect(int p, int length) { int n = 2 * length; p = ((p % n) + n) % n; return p < length ? p : n - 1 - p; }
    private static void Fft2(Complex[] values, int size, bool inverse)
    {
        var line = new Complex[size];
        for (int y = 0; y < size; y++) { Array.Copy(values, y * size, line, 0, size); Fft(line, inverse); Array.Copy(line, 0, values, y * size, size); }
        for (int x = 0; x < size; x++) { for (int y = 0; y < size; y++) line[y] = values[y * size + x]; Fft(line, inverse); for (int y = 0; y < size; y++) values[y * size + x] = line[y]; }
    }
    private static void Fft(Complex[] a, bool inverse)
    {
        for (int i = 1, j = 0; i < a.Length; i++) { int bit = a.Length >> 1; for (; (j & bit) != 0; bit >>= 1) j ^= bit; j ^= bit; if (i < j) (a[i], a[j]) = (a[j], a[i]); }
        for (int length = 2; length <= a.Length; length <<= 1)
        {
            Complex root = Complex.FromPolarCoordinates(1, (inverse ? 2 : -2) * Math.PI / length);
            for (int start = 0; start < a.Length; start += length)
            {
                Complex factor = Complex.One;
                for (int j = 0; j < length / 2; j++) { var u = a[start + j]; var v = a[start + j + length / 2] * factor; a[start + j] = u + v; a[start + j + length / 2] = u - v; factor *= root; }
            }
        }
        if (inverse) for (int i = 0; i < a.Length; i++) a[i] /= a.Length;
    }
}
