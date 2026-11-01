using System.Text.Json.Nodes;
using OpenRevelare.Core;
using Xunit;

namespace OpenRevelare.Tests;

public sealed class NewtonRingsTests
{
    // Held-out phase functions exercise local direction changes and non-concentric contours.
    // Compare to the known clean texture, not a blur that could erase the picture with the rings.
    [Theory]
    [InlineData("linear")]
    [InlineData("ellipse")]
    [InlineData("warped")]
    [InlineData("chirp")]
    [InlineData("mono")]
    [InlineData("saddle")]
    [InlineData("crossed")]
    public void Reduces_varied_fringe_shapes_without_replacing_scene_texture(string shape)
    {
        var clean = Scene(192, 160);
        var corrupted = AddFringe(clean, shape);
        var p = new FrameParams { NewtonEnabled = true, NewtonStrength = 1, NewtonLuminance = 1,
            NewtonField = NewtonRings.Analyse(corrupted, .6), NewtonAutomatic = true };
        var repaired = Copy(corrupted);
        NewtonRings.Apply(repaired, p);
        double before = Error(clean, corrupted), after = Error(clean, repaired);
        Assert.True(after < before * .92, $"{shape}: error before={before:G5}, after={after:G5}, coverage={p.NewtonField.Coverage:P1}");
        Assert.All(repaired.Data, v => Assert.True(float.IsFinite(v) && v > 0));
    }

    [Fact]
    public void Smooth_clean_scene_is_unchanged_when_no_spectral_candidates_exist()
    {
        var clean = new ImageBuffer(128, 96); Array.Fill(clean.Data, .35f);
        var field = NewtonRings.Analyse(clean);
        Assert.Equal(0, field.Coverage);
        var result = Copy(clean);
        NewtonRings.Apply(result, new() { NewtonEnabled = true, NewtonAutomatic = true, NewtonField = field });
        Assert.Equal(clean.Data, result.Data);
    }

    [Fact]
    public void Natural_colour_steps_are_not_treated_as_rainbow_fringes()
    {
        var image = new ImageBuffer(192, 160);
        for (int y = 0; y < image.Height; y++) for (int x = 0; x < image.Width; x++)
        {
            float value = x < 80 ? .2f : .6f;
            image.Data[(y * image.Width + x) * 3] = value;
            image.Data[(y * image.Width + x) * 3 + 1] = .15f + value * .3f;
            image.Data[(y * image.Width + x) * 3 + 2] = .1f + value * .2f;
        }
        Assert.InRange(NewtonRings.Analyse(image).Coverage, 0, .10);
    }

    [Fact]
    public void Arbitrary_painted_union_and_exclusions_preserve_every_unselected_pixel()
    {
        var source = AddFringe(Scene(128, 100), "warped");
        var p = new FrameParams { NewtonEnabled = true, NewtonStrength = 1, NewtonField = NewtonRings.Analyse(source),
            NewtonStrokes = new() { new(.35,.3,.12), new(.4,.4,.12), new(.48,.5,.12), new(.4,.4,.04,true) } };
        var mask = NewtonRings.Mask(source.Width, source.Height, p.NewtonField, false, p.NewtonStrokes);
        Assert.Equal(0, mask[40 * 128 + 51]);
        var repaired = Copy(source); NewtonRings.Apply(repaired, p);
        for (int i = 0; i < mask.Length; i++) if (mask[i] == 0)
            for (int c = 0; c < 3; c++) Assert.Equal(source.Data[3 * i + c], repaired.Data[3 * i + c]);
        p.NewtonStrength = 0; var disabled = Copy(source); NewtonRings.Apply(disabled, p);
        Assert.Equal(source.Data, disabled.Data);
    }

    [Fact]
    public void Field_and_brushes_round_trip_and_clone_without_sharing_mutable_edits()
    {
        var field = NewtonRings.Analyse(AddFringe(Scene(80,64), "linear"));
        var p = new FrameParams { NewtonEnabled = true, NewtonField = field, NewtonStrokes = new() { new(.5,.5,.1) } };
        var clone = p.Clone(); clone.NewtonStrokes.Clear(); Assert.Single(p.NewtonStrokes);
        string dir = Path.Combine(TestDataIsolation.Root, "newton-" + Guid.NewGuid().ToString("N")); Directory.CreateDirectory(dir);
        string project = Path.Combine(dir, "roll.ncproj");
        Project.Save(project, new Project.Data { Frames = new() { new() { SourcePath = "test.tif", Params = p } } });
        Assert.Equal(5, JsonNode.Parse(File.ReadAllText(project))!["version"]!.GetValue<int>());
        var loaded = Project.Load(project).Frames[0].Params;
        Assert.True(loaded.NewtonEnabled); Assert.Equal(p.NewtonStrokes, loaded.NewtonStrokes);
        for (int c = 0; c < 5; c++) Assert.Equal(field.Sample(.5,.5,c), loaded.NewtonField!.Sample(.5,.5,c));
        var a = AddFringe(Scene(80,64), "linear"); var b = Copy(a);
        NewtonRings.Apply(a,p); NewtonRings.Apply(b,loaded);
        Assert.Equal(a.Data,b.Data);
        Assert.ThrowsAny<Exception>(() => NewtonField.Decode("invalid-payload"));
    }

    [Fact]
    public void Regional_and_full_grid_repairs_match_without_redetecting_in_the_slice()
    {
        var source = AddFringe(Scene(160,128), "chirp");
        var p = new FrameParams { NewtonEnabled = true, NewtonAutomatic = true, NewtonField = NewtonRings.Analyse(source), OutputIntent = OutputIntent.None };
        var full = Copy(source); NewtonRings.Apply(full, p);
        var rect = (.25,.25,.5,.5); var slice = Geometry.ApplyCrop(source, rect);
        NewtonRings.Apply(slice, p, rect);
        Assert.Equal(Geometry.ApplyCrop(full, rect).Data, slice.Data);
        var original = (float[])source.Data.Clone(); Pipeline.ProcessFrame(source, p);
        Assert.Equal(original, source.Data);
        var roi = new RegionRender.Roi(.3,.3,.3,.3);
        var region = RegionRender.Render(source, p, roi);
        var expected = Pipeline.ProcessFrame(source, p);
        var actual = Geometry.ApplyCrop(expected, (region.Realised.X, region.Realised.Y, region.Realised.W, region.Realised.H));
        Assert.Equal(actual.Data, region.Image.Data);
    }

    internal static ImageBuffer Scene(int w, int h)
    {
        var image = new ImageBuffer(w,h);
        for(int y=0;y<h;y++) for(int x=0;x<w;x++) for(int c=0;c<3;c++)
            image.Data[(y*w+x)*3+c] = (float)(.35 + .025*Math.Sin(x*.6+y*.9) + .08*x/w + .02*c);
        return image;
    }
    internal static ImageBuffer AddFringe(ImageBuffer clean, string shape)
    {
        var result = Copy(clean);
        for(int y=0;y<clean.Height;y++) for(int x=0;x<clean.Width;x++)
        {
            double phase = shape switch {
                "ellipse" => Math.Sqrt(Math.Pow((x-71)*.7,2)+Math.Pow((y-63)*1.2,2))*.52,
                "warped" => x*.30 + 2.5*Math.Sin(y*.038) + .0006*(x-90)*(y-70),
                "chirp" => x*.18 + .0015*x*x + y*.08,
                "saddle" => .002*(x-70)*(x-70) - .0015*(y-60)*(y-60) + x*.2,
                _ => x*.38 + y*.12 };
            for(int c=0;c<3;c++) result.Data[(y*clean.Width+x)*3+c] *= (float)Math.Exp(.09*Math.Sin(phase + (shape=="mono" ? 0 : c*2.1))
                    + (shape=="crossed" ? .04*Math.Sin(x*.18-y*.32+c*1.6) : 0));
        }
        return result;
    }
    private static ImageBuffer Copy(ImageBuffer image) => new(image.Width,image.Height,(float[])image.Data.Clone());
    private static double Error(ImageBuffer a, ImageBuffer b) => a.Data.Zip(b.Data).Average(v => Math.Pow(v.First-v.Second,2));
}
