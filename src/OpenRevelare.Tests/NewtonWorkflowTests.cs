using System.Security.Cryptography;
using OpenRevelare.Core;
using OpenRevelare.Gui.ViewModels;
using Xunit;

namespace OpenRevelare.Tests;

public sealed class NewtonWorkflowTests
{
    // These exercise the real live-parameter/undo/project path, not just the filter. Painted
    // coordinates must survive crop, straighten and mirrors before the field is generated.
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task Automatic_and_painted_repairs_are_undoable_saved_and_do_not_edit_the_source(bool automatic)
    {
        string directory = Path.Combine(TestDataIsolation.Root, "newton-workflow-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        string source = Path.Combine(directory, "fringes.tif"), project = Path.Combine(directory, "roll.ncproj");
        TiffIO.ExportTiff16(NewtonRingsTests.AddFringe(NewtonRingsTests.Scene(192,160), "warped"), source);
        byte[] original = SHA256.HashData(File.ReadAllBytes(source));
        Project.Save(project, new Project.Data
        {
            Meta = new() { InputType = "tiff", TiffIsLinear = true },
            Frames = new() { new() { SourcePath = source } },
        });
        using var vm = new MainViewModel();
        await vm.OpenProjectAsync(project);
        for (int i = 0; i < 400 && (vm.IsBusy || vm.RollAnalysisPending); i++) await Task.Delay(25);
        Assert.True(vm.HasImage); Assert.False(vm.IsBusy);
        vm.SetCrop((.15,.1,.65,.75)); vm.RotateCw(); vm.Rotation = 3; vm.FlipHorizontal();
        if (!automatic)
        {
            Assert.True(vm.PaintNewtonAt(.3,.4,false));
            Assert.True(vm.PaintNewtonAt(.4,.5,false));
            Assert.True(vm.PaintNewtonAt(.5,.6,false));
            vm.CommitNewtonStroke();
        }
        await vm.AnalyseNewtonAsync(automatic);
        Assert.False(vm.IsNewtonAnalysing);
        Assert.True(vm.NewtonEnabled, vm.StatusText);
        Assert.True(vm.CurrentFrame!.Params.NewtonField is not null, "field after analysis: " + vm.StatusText);
        NewtonField field = vm.CurrentFrame.Params.NewtonField!;
        Assert.Equal(automatic, vm.CurrentFrame.Params.NewtonAutomatic);
        Assert.Equal(automatic ? 0 : 3, vm.CurrentFrame.Params.NewtonStrokes.Count);
        vm.Undo(); Assert.Null(vm.CurrentFrame.Params.NewtonField); Assert.False(vm.NewtonEnabled);
        vm.Redo(); Assert.True(vm.CurrentFrame.Params.NewtonField is not null, "field after redo"); Assert.True(vm.NewtonEnabled);
        vm.NewtonStrength = .55;
        Assert.True(await vm.SaveProjectAsync(project));
        var loaded = Project.Load(project).Frames[0].Params;
        Assert.True(loaded.NewtonEnabled); Assert.Equal(.55, loaded.NewtonStrength);
        Assert.Equal(vm.CurrentFrame.Params.NewtonStrokes, loaded.NewtonStrokes);
        for (int c = 0; c < 5; c++) Assert.InRange(Math.Abs(field.Sample(.5,.5,c) - loaded.NewtonField!.Sample(.5,.5,c)),0,1d/16384);
        Assert.Equal(original, SHA256.HashData(File.ReadAllBytes(source)));
        vm.ClearNewton(); Assert.False(vm.NewtonEnabled); Assert.Null(vm.CurrentFrame.Params.NewtonField);
        vm.Undo(); Assert.True(vm.NewtonEnabled); Assert.True(vm.CurrentFrame.Params.NewtonField is not null, "field after undo clear");
        var previous = vm.CurrentFrame.Params.NewtonField;
        vm.PropertyChanged += (_, e) => { if (e.PropertyName == nameof(vm.StatusText) && vm.IsNewtonAnalysing) vm.CancelNewtonAnalysis(); };
        await vm.AnalyseNewtonAsync(true);
        Assert.Same(previous, vm.CurrentFrame.Params.NewtonField);
        Assert.False(vm.IsNewtonAnalysing);
    }
}
