using System.Security.Cryptography;
using System.Text.Json;
using OpenRevelare.Core;
using OpenRevelare.DeepWb.Onnx;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

// Deliberately takes no painted mask or annotation: those belong to evaluation, not inference.
if (args.Length < 3)
{
    Console.Error.WriteLine("Usage: <output-directory> <project.ncproj> <RAW...> [--ai-model=<path>]");
    return 2;
}
string output = Path.GetFullPath(args[0]); Directory.CreateDirectory(output);
var project = Project.Load(args[1]);
string? modelPath = args.FirstOrDefault(a => a.StartsWith("--ai-model="))?[11..];
using var ai = modelPath is null ? null : new OnnxNewtonEstimator(modelPath);
string? modelSha = modelPath is null ? null : Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(modelPath))).ToLowerInvariant();
string projectSha = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(args[1]))).ToLowerInvariant();
var records = new List<object>();
foreach (string path in args.Skip(2).Where(a => !a.StartsWith("--")))
{
    string name = Path.GetFileNameWithoutExtension(path);
    var parameters = project.Frames.FirstOrDefault(f => Path.GetFileName(f.SourcePath) == Path.GetFileName(path))?.Params.Clone()
        ?? throw new ArgumentException("Frame missing from project: " + name);
    parameters.NewtonEnabled = false;
    var (images, sourceWidth, sourceHeight) = RawDecode.DecodeRawDownsampled(path, RawDecode.FbddMode.Off, new[] { 4096 }, RawDecode.Demosaic.Full);
    ImageBuffer source = images[0];
    var linearView = Warp(source, parameters);
    WriteRgb(Path.Combine(output, name + ".rgbf"), linearView);
    SaveJpeg(Path.Combine(output, name + ".jpg"), Pipeline.ProcessFrame(source, parameters));
    var legacy = NewtonRings.Analyse(source, .5);
    WriteRgb(Path.Combine(output, name + ".legacy.rgbf"), Warp(FieldImage(legacy, source.Width, source.Height), parameters));
    // Use only the existing photographic crop, not the user's two Newton-band annotations.
    // For this diagnostic the source files have no quarter-turns or flips; reject instead of
    // guessing an inverse crop and accidentally measuring a different part of the source.
    if (parameters.QuarterTurns != 0 || parameters.FlipH || parameters.FlipV)
        throw new NotSupportedException("Diagnostic region mapping currently requires an unmirrored project frame.");
    var crop = parameters.CropRect ?? (0d, 0d, 1d, 1d);
    int x0 = Math.Max(0, (int)Math.Floor(crop.Item1 * sourceWidth));
    int y0 = Math.Max(0, (int)Math.Floor(crop.Item2 * sourceHeight));
    int x1 = Math.Min(sourceWidth, (int)Math.Ceiling((crop.Item1 + crop.Item3) * sourceWidth));
    int y1 = Math.Min(sourceHeight, (int)Math.Ceiling((crop.Item2 + crop.Item4) * sourceHeight));
    var region = RawDecode.DecodeRawRegion(path, RawDecode.RawBackend.LibRaw, RawDecode.FbddMode.Off,
        x0, y0, x1-x0, y1-y0, sourceWidth, sourceHeight) ?? throw new NotSupportedException("No RAW regional decoder.");
    var rect = (region.X0/(double)sourceWidth, region.Y0/(double)sourceHeight,
        region.Slice.Width/(double)sourceWidth, region.Slice.Height/(double)sourceHeight);
    var bands = NewtonRings.AnalyseBands(region.Slice, rect, -parameters.Rotation * Math.PI / 180);
    WriteRgb(Path.Combine(output, name + ".bands.rgbf"), Warp(FieldImage(bands, source.Width, source.Height), parameters));
    if (ai is not null)
        WriteRgb(Path.Combine(output, name + ".ai.rgbf"), Warp(FieldImage(ai.Estimate(source,.5), source.Width, source.Height), parameters));
    using var stream = File.OpenRead(path);
    records.Add(new { File = Path.GetFileName(path), SourceSha256 = Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant(),
        sourceWidth, sourceHeight, ViewWidth = linearView.Width, ViewHeight = linearView.Height,
        AnalysisReceivesAnnotation = false, ProjectSha256 = projectSha, AiModelSha256 = modelSha, LegacyCoverage = legacy.Coverage, BandsCoverage = bands.Coverage });
    Console.WriteLine(name + " detection export complete");
}
File.WriteAllText(Path.Combine(output,"sources.json"),JsonSerializer.Serialize(records,new JsonSerializerOptions { WriteIndented = true }));
return 0;

static ImageBuffer Warp(ImageBuffer source, FrameParams p)
{
    var image = p.DistortionK1 == 0 ? source : LensCorrections.ApplyDistortion(source,p.DistortionK1);
    image = Geometry.ApplyOrientation(image,p.QuarterTurns,p.FlipH,p.FlipV);
    image = Geometry.ApplyRotation(image,p.Rotation,0);
    return p.CropRect is {} crop ? Geometry.ApplyCrop(image,crop) : image;
}
static ImageBuffer FieldImage(NewtonField field, int width, int height)
{
    var image = new ImageBuffer(width,height);
    for(int y=0;y<height;y++) for(int x=0;x<width;x++)
    {
        float confidence = field.Sample((x+.5)/width,(y+.5)/height,4);
        for(int c=0;c<3;c++) image.Data[(y*width+x)*3+c]=confidence;
    }
    return image;
}
static void WriteRgb(string path, ImageBuffer image)
{
    using var writer=new BinaryWriter(File.Create(path));
    writer.Write(image.Width); writer.Write(image.Height);
    foreach(float value in image.Data) writer.Write(value);
}
static void SaveJpeg(string path, ImageBuffer image)
{
    using var output=new Image<Rgb24>(image.Width,image.Height);
    for(int y=0;y<image.Height;y++) for(int x=0;x<image.Width;x++)
    {
        int i=(y*image.Width+x)*3;
        byte Byte(int c)=>(byte)Math.Round(Math.Clamp(image.Data[i+c],0,1)*255);
        output[x,y]=new(Byte(0),Byte(1),Byte(2));
    }
    output.SaveAsJpeg(path);
}
