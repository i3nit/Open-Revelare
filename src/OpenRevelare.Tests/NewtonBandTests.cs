using System.Text.Json.Nodes;
using OpenRevelare.Core;
using Xunit;

namespace OpenRevelare.Tests;

public sealed class NewtonBandTests
{
    [Theory]
    [InlineData(0, false, false)]
    [InlineData(0, false, true)]
    [InlineData(0, true, false)]
    [InlineData(1.5707963267948966, false, false)]
    [InlineData(.21, false, false)]
    public void Picture_region_resolves_fine_curved_bands_for_both_transmission_polarities(double direction, bool negative, bool monochrome)
    {
        // Whole-scan boxing aliased these 3–4 pixel bands before the old detector could see them.
        // A separately decoded picture grid must remove the band without changing the carrier detail.
        const int w = 256, h = 224;
        var clean = NewtonRingsTests.Scene(w, h);
        if (negative) for (int i = 0; i < clean.Data.Length; i++) clean.Data[i] = .8f - clean.Data[i];
        var observed = new ImageBuffer(w, h, (float[])clean.Data.Clone());
        for (int y = 0; y < h; y++) for (int x = 0; x < w; x++)
        {
            double normal = -x * Math.Sin(direction) + y * Math.Cos(direction);
            double along = x * Math.Cos(direction) + y * Math.Sin(direction);
            double phase = normal * 1.8 + 2 * Math.Sin(along * .025);
            for (int c = 0; c < 3; c++) observed.Data[(y * w + x) * 3 + c] *= (float)Math.Exp(.065 * Math.Sin(phase + (monochrome ? 0 : c * 2.1)));
        }
        var field = NewtonRings.AnalyseBands(observed, (0, 0, 1, 1), direction);
        var repaired = new ImageBuffer(w, h, (float[])observed.Data.Clone());
        NewtonRings.Apply(repaired, new() { NewtonEnabled = true, NewtonStrength = 1, NewtonLuminance = 1,
            NewtonField = field, NewtonAutomatic = true });
        double Error(ImageBuffer image) => image.Data.Zip(clean.Data).Average(pair => Math.Pow(Math.Log(pair.First / pair.Second), 2));
        Assert.True(Error(repaired) < Error(observed) * .8, $"direction={direction}, before={Error(observed)}, after={Error(repaired)}");
    }

    [Fact]
    public void Local_field_sampling_and_project_roundtrip_retain_source_phase_and_compatibility_floor()
    {
        var residual = new ImageBuffer(24, 16);
        for (int i = 0; i < residual.Data.Length; i++) residual.Data[i] = (float)(.05 * Math.Sin(i * .17));
        var local = NewtonField.FromLogResidual(residual, Enumerable.Repeat(.8f, residual.PixelCount).ToArray());
        var rect = (.2, .3, .4, .5); var field = local.InSourceRect(rect);
        Assert.Equal(0, field.Sample(.1, .5, 0)); Assert.Equal(0, field.Sample(.8, .5, 4));
        for (int c = 0; c < 5; c++) Assert.Equal(local.Sample(.25, .5, c), field.Sample(.3, .55, c), 6);
        string directory = Path.Combine(TestDataIsolation.Root, "newton-band-" + Guid.NewGuid()); Directory.CreateDirectory(directory);
        string path = Path.Combine(directory, "roll.ncproj");
        var p = new FrameParams { NewtonEnabled = true, NewtonAutomatic = true, NewtonField = field, OutputIntent = OutputIntent.None };
        var project = new Project.Data { Frames = new() { new() { SourcePath = "bands.tif", Params = p } } };
        Project.Save(path, project);
        Assert.Equal(6, JsonNode.Parse(File.ReadAllText(path))!["version"]!.GetValue<int>());
        var loaded = Project.Load(path).Frames[0].Params;
        Assert.Equal(rect, loaded.NewtonField!.SourceRect);
        var source = NewtonRingsTests.Scene(240, 160);
        var full = new ImageBuffer(source.Width, source.Height, (float[])source.Data.Clone()); NewtonRings.Apply(full, p);
        var reopened = new ImageBuffer(source.Width, source.Height, (float[])source.Data.Clone()); NewtonRings.Apply(reopened, loaded);
        Assert.Equal(full.Data, reopened.Data);
        var slice = Geometry.ApplyCrop(source, rect); NewtonRings.Apply(slice, p, rect);
        Assert.Equal(Geometry.ApplyCrop(full, rect).Data, slice.Data);
        p.NewtonField = local; Project.Save(path, project);
        Assert.Equal(5, JsonNode.Parse(File.ReadAllText(path))!["version"]!.GetValue<int>());
        Assert.Throws<ArgumentException>(() => local.InSourceRect((double.NaN,0,1,1)));
        Assert.Throws<ArgumentException>(() => local.InSourceRect((.5,0,1,1)));
    }

    [Fact]
    public void Boxing_does_not_stretch_an_incomplete_cell_back_into_the_band_phase()
    {
        var image = new ImageBuffer(1031, 129); Array.Fill(image.Data, .3f);
        var field = NewtonRings.AnalyseBands(image, (.2,.3,.5,.2));
        Assert.Equal(515, field.Width); Assert.Equal(64, field.Height);
        Assert.Equal(.5 * 1030 / 1031, field.SourceRect.W, 12);
        Assert.Equal(.2 * 128 / 129, field.SourceRect.H, 12);
        Assert.Equal(0d, field.Coverage);
    }
}
