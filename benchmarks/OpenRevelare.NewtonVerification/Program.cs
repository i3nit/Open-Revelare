using System.Diagnostics;
using System.Text.Json;
using OpenRevelare.Core;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

if (args.Length < 3) { Console.Error.WriteLine("Usage: <output-directory> <ncproj-file-or-dash> <RAW> [RAW...] [--manual]"); return 2; }
string output = Path.GetFullPath(args[0]); Directory.CreateDirectory(output);
Project.Data? project = args[1] == "-" ? null : Project.Load(args[1]);
var report = new List<object>();
bool manual = args.Contains("--manual");
foreach (string path in args.Skip(2).Where(a => a != "--manual"))
{
    var timer = Stopwatch.StartNew();
    var (images, w, h) = RawDecode.DecodeRawDownsampled(path, RawDecode.FbddMode.Off, new[] { 1600 }, RawDecode.Demosaic.Preview);
    var pixels = images[0];
    var p = project?.Frames.FirstOrDefault(f => Path.GetFileName(f.SourcePath) == Path.GetFileName(path))?.Params.Clone() ?? new FrameParams();
    p.NewtonEnabled = false;
    var before = Pipeline.ProcessFrame(pixels, p);
    var field = NewtonRings.Analyse(pixels, .5, manual: manual);
    p.NewtonEnabled = true; p.NewtonField = field; p.NewtonAutomatic = !manual; p.NewtonStrength = .8; p.NewtonLuminance = .25;
    if (manual)
    {
        p.NewtonStrokes = new();
        // A whole-source stress test, not a recommendation to select the complete photograph.
        for (double y = .05; y < 1; y += .1) for (double x = .05; x < 1; x += .1)
            p.NewtonStrokes.Add(new NewtonStroke(x,y,.14));
    }
    var after = Pipeline.ProcessFrame(pixels, p);
    var mask = NewtonRings.Mask(pixels.Width, pixels.Height, field, !manual, p.NewtonStrokes);
    var flags = new ImageBuffer(pixels.Width, pixels.Height);
    for (int i=0;i<mask.Length;i++) for (int c=0;c<3;c++) flags.Data[3*i+c]=mask[i];
    if(p.DistortionK1!=0) flags=LensCorrections.ApplyDistortion(flags,p.DistortionK1);
    flags=Geometry.ApplyOrientation(flags,p.QuarterTurns,p.FlipH,p.FlipV);
    flags=Geometry.ApplyRotation(flags,p.Rotation,0);
    if(p.CropRect is {} crop) flags=Geometry.ApplyCrop(flags,crop);
    using var comparison = new Image<Rgb24>(before.Width*3,before.Height);
    double delta = 0;
    for(int y=0;y<before.Height;y++) for(int x=0;x<before.Width;x++)
    {
        int i=(y*before.Width+x)*3;
        byte Channel(ImageBuffer image,int c) => (byte)(Math.Clamp(image.Data[i+c],0,1)*255);
        var a = new Rgb24(Channel(before,0),Channel(before,1),Channel(before,2));
        var b = new Rgb24(Channel(after,0),Channel(after,1),Channel(after,2));
        comparison[x,y]=a; comparison[x+before.Width,y]=b;
        float alpha=Math.Clamp(flags.Data[i]*.55f,0,1);
        comparison[x+before.Width*2,y]=new Rgb24((byte)(a.R*(1-alpha)),(byte)(a.G*(1-alpha)+180*alpha),(byte)(a.B*(1-alpha)+255*alpha));
        for(int c=0;c<3;c++) delta+=Math.Abs(before.Data[i+c]-after.Data[i+c]);
    }
    string name=Path.GetFileNameWithoutExtension(path);
    comparison.SaveAsJpeg(Path.Combine(output,name+"-comparison.jpg"));
    File.WriteAllText(Path.Combine(output,name+"-field.txt"),field.Encode());
    report.Add(new { File=Path.GetFileName(path), SourceWidth=w, SourceHeight=h, ManualWholeSourceStressTest=manual, field.Coverage,
        MeanAbsoluteRenderedChange=delta/before.Data.Length, Milliseconds=timer.ElapsedMilliseconds });
    Console.WriteLine($"{name}: candidate={field.Coverage:P1}, mean change={delta/before.Data.Length:G4}, {timer.ElapsedMilliseconds} ms");
}
File.WriteAllText(Path.Combine(output,"report.json"),JsonSerializer.Serialize(report,new JsonSerializerOptions{WriteIndented=true}));
return 0;
