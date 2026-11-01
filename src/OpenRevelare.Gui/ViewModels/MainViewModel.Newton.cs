using Avalonia.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using OpenRevelare.Core;
using OpenRevelare.Gui.Interop;
using OpenRevelare.Presentation;

namespace OpenRevelare.Gui.ViewModels;

public partial class MainViewModel
{
    [ObservableProperty] private bool _newtonEnabled;
    [ObservableProperty] private bool _showNewtonMask;
    [ObservableProperty] private double _newtonStrength = .8;
    [ObservableProperty] private double _newtonLuminance = .25;
    [ObservableProperty] private double _newtonSensitivity = .5;
    [ObservableProperty] private double _newtonBrushSize = .04;
    [ObservableProperty] private Bitmap? _newtonMaskOverlay;
    [ObservableProperty] private PresentationScene? _newtonMaskScene;
    private NewtonField? _newtonField;
    private bool _newtonAutomatic;
    private List<NewtonStroke> _newtonStrokes = new();
    private CancellationTokenSource? _newtonCts;
    public bool IsNewtonAnalysing => _newtonCts is not null;
    public string NewtonSummary => _newtonField is null
        ? Loc.F($"手绘采样 { _newtonStrokes.Count } · 尚未分析")
        : Loc.F($"候选覆盖 {_newtonField.Coverage:P0} · 手绘采样 {_newtonStrokes.Count}");

    partial void OnNewtonEnabledChanged(bool value) => ScheduleRender();
    partial void OnNewtonStrengthChanged(double value) => ScheduleRender();
    partial void OnNewtonLuminanceChanged(double value) => ScheduleRender();
    partial void OnShowNewtonMaskChanged(bool value) { UpdateNewtonOverlay(); InvalidatePresentation(); }
    partial void OnNewtonMaskSceneChanged(PresentationScene? value) => InvalidatePresentation();
    partial void OnNewtonMaskOverlayChanging(Bitmap? oldValue, Bitmap? newValue)
    { if (!ReferenceEquals(oldValue, newValue)) Retire(oldValue); }

    private void LoadNewton(FrameParams p)
    {
        _newtonField = p.NewtonField; _newtonAutomatic = p.NewtonAutomatic;
        _newtonStrokes = new List<NewtonStroke>(p.NewtonStrokes);
        NewtonEnabled = p.NewtonEnabled; NewtonStrength = p.NewtonStrength; NewtonLuminance = p.NewtonLuminance;
        OnPropertyChanged(nameof(NewtonSummary));
        UpdateNewtonOverlay();
    }

    public bool PaintNewtonAt(double displayedX, double displayedY, bool erase)
    {
        if (DisplayPointToNewtonSource(displayedX, displayedY) is not { } point) return false;
        double x = point.X, y = point.Y;
        if (_newtonStrokes.Count >= 50000) return false;
        if (_newtonStrokes.LastOrDefault() is var last && last.Radius > 0 && last.Erase == erase
            && NewtonSourceDistance((x, y), (last.X, last.Y)) < NewtonBrushSize * .2) return false;
        _newtonStrokes.Add(new NewtonStroke(x, y, NewtonBrushSize, erase));
        return true;
    }

    private (double X, double Y)? DisplayPointToNewtonSource(double displayedX, double displayedY)
    {
        if (DisplayPointToDustSource(displayedX, displayedY) is not { } position || _previewLinear is null) return null;
        var box = _previewMargin ?? (0d, 0d, 1d, 1d);
        // The repair is attached to the sensor grid, before optical correction. Undo that
        // correction's backward map as well as the crop/rotation handled by the dust mapper.
        double x = (position.X - box.Item1) / box.Item3, y = (position.Y - box.Item2) / box.Item4;
        double w = _previewLinear.Width, h = _previewLinear.Height;
        double cx = (w - 1) / 2, cy = (h - 1) / 2, norm = Math.Min(cx, cy);
        double px = x * w - .5, py = y * h - .5;
        double dx = (px - cx) / Math.Max(norm, 1), dy = (py - cy) / Math.Max(norm, 1);
        double scale = 1 + DistortionK1 * (dx * dx + dy * dy);
        x = box.Item1 + Math.Clamp((cx + (px - cx) * scale + .5) / w, 0, 1) * box.Item3;
        y = box.Item2 + Math.Clamp((cy + (py - cy) * scale + .5) / h, 0, 1) * box.Item4;
        return (x, y);
    }

    private double NewtonSourceDistance((double X, double Y) a, (double X, double Y) b)
    {
        if (_previewLinear is null) return double.MaxValue;
        var box = _previewMargin ?? (0d, 0d, 1d, 1d);
        double w = _previewLinear.Width / box.Item3, h = _previewLinear.Height / box.Item4;
        double min = Math.Max(1, Math.Min(w, h));
        return Math.Sqrt(Math.Pow((a.X - b.X) * w / min, 2) + Math.Pow((a.Y - b.Y) * h / min, 2));
    }

    public void CommitNewtonStroke()
    {
        MarkEdit(); UpdateNewtonOverlay(); OnPropertyChanged(nameof(NewtonSummary));
        ScheduleRender(); CommitUndo();
    }

    public void ClearNewton()
    {
        CommitUndo(); _newtonField = null; _newtonAutomatic = false; _newtonStrokes.Clear();
        NewtonEnabled = false; MarkEdit(); UpdateNewtonOverlay(); OnPropertyChanged(nameof(NewtonSummary));
        ScheduleRender(); CommitUndo();
    }
    public void CancelNewtonAnalysis() => _newtonCts?.Cancel();

    public async Task AnalyseNewtonAsync(bool automatic)
    {
        if (CurrentFrame is not { } frame || !HasImage || IsBusy || _newtonCts is not null) return;
        if (!automatic && !_newtonStrokes.Any(s => !s.Erase))
        { StatusText = Loc.T("请先用牛顿环区域笔涂抹需要修复的位置。"); return; }
        CommitLiveParams(frame); CommitUndo();
        var roll = _roll; int version = _editVersion;
        double sensitivity = NewtonSensitivity;
        var cts = new CancellationTokenSource(); _newtonCts = cts; IsBusy = true;
        OnPropertyChanged(nameof(IsNewtonAnalysing));
        try
        {
            StatusText = Loc.T("正在分析局部干涉纹频谱 …");
            var entry = await PreviewAsync(frame.Path, null).WaitAsync(cts.Token);
            var field = await Task.Run(() => NewtonRings.Analyse(entry.Preview, sensitivity, cts.Token, manual: !automatic), cts.Token);
            cts.Token.ThrowIfCancellationRequested();
            if (_disposed || CurrentFrame != frame || !ReferenceEquals(_roll, roll) || _editVersion != version)
            { StatusText = Loc.T("牛顿环分析未应用：图像或参数已改变，请重试。"); return; }
            if (field.Coverage == 0)
            { StatusText = Loc.T("未找到可靠的周期干涉纹，原修复保持不变；可手绘区域或提高检测灵敏度后重试。"); return; }
            _newtonField = field; _newtonAutomatic = automatic;
            NewtonEnabled = true; ShowNewtonMask = true;
            OnPropertyChanged(nameof(NewtonSummary)); UpdateNewtonOverlay(); ScheduleRender(); CommitUndo();
            StatusText = automatic ? Loc.F($"已生成牛顿环候选并修复，候选覆盖 {field.Coverage:P0}；请检查误选并用排除笔调整。")
                : Loc.T("已修复手绘区域；可调整强度、亮度修复比例或关闭比较。");
        }
        catch (OperationCanceledException) { StatusText = Loc.T("牛顿环分析已取消，原修复保持不变。"); }
        catch (Exception ex) { StatusText = Loc.T("牛顿环分析失败：") + ex.Message; }
        finally
        {
            _newtonCts = null; cts.Dispose();
            OnPropertyChanged(nameof(IsNewtonAnalysing));
            if (ReferenceEquals(_roll, roll)) IsBusy = false;
        }
    }

    private void UpdateNewtonOverlay()
    {
        if (!ShowNewtonMask || _previewLinear is null || (_newtonField is null && _newtonStrokes.Count == 0))
        { NewtonMaskOverlay = null; NewtonMaskScene = null; return; }
        var p = ForPreview(BuildParams());
        var mask = NewtonRings.Mask(_previewLinear.Width, _previewLinear.Height, _newtonField, _newtonAutomatic, _newtonStrokes, p.NewtonSourceRect);
        var flags = new ImageBuffer(_previewLinear.Width, _previewLinear.Height);
        for (int i = 0; i < mask.Length; i++) for (int c = 0; c < 3; c++) flags.Data[i * 3 + c] = mask[i];
        if (p.DistortionK1 != 0) flags = LensCorrections.ApplyDistortion(flags, p.DistortionK1);
        flags = Geometry.ApplyOrientation(flags, p.QuarterTurns, p.FlipH, p.FlipV);
        flags = Geometry.ApplyRotation(flags, p.Rotation, fill: 0);
        if (p.CropRect is { } crop) flags = Geometry.ApplyCrop(flags, crop);
        bool[] shaped = Enumerable.Range(0, flags.PixelCount).Select(i => flags.Data[i * 3] > .1f).ToArray();
        // The scene is also consumed by non-windowed view-model hosts. The managed bitmap
        // fallback needs an initialized Avalonia application; it is not part of the repair.
        NewtonMaskOverlay = Avalonia.Application.Current is null ? null
            : BitmapConvert.ToMaskOverlay(shaped, flags.Width, flags.Height, r: 40, g: 200, b: 255, a: 115);
        NewtonMaskScene = BuildMaskPresentationScene(shaped, flags.Width, flags.Height, red: 40, green: 200, blue: 255, alpha: 115);
    }
}
