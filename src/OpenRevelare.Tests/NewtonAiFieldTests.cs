using OpenRevelare.Core;
using OpenRevelare.DeepWb.Onnx;
using Xunit;

namespace OpenRevelare.Tests;

public sealed class NewtonAiFieldTests
{
    [Fact]
    public void Learned_log_residual_is_bounded_immutable_and_round_trips_without_new_rounding()
    {
        var residual = new ImageBuffer(2, 2);
        for (int i = 0; i < 4; i++) { residual.Data[i * 3] = .15f; residual.Data[i * 3 + 1] = -.06f; residual.Data[i * 3 + 2] = .03f; }
        float[] confidence = { .9f, .9f, .9f, .9f };
        var field = NewtonField.FromLogResidual(residual, confidence);
        var restored = NewtonField.Decode(field.Encode());
        Assert.NotNull(restored);
        Array.Fill(residual.Data, 1); Array.Fill(confidence, 0);
        Assert.Equal(1d, field.Coverage);
        for (int c = 0; c < 5; c++) Assert.Equal(field.Sample(.25,.25,c), restored.Sample(.25,.25,c));
        Assert.InRange(field.Sample(.25,.25,0) + field.Sample(.25,.25,3), .1498, .1502);
        var source = new ImageBuffer(2, 2); Array.Fill(source.Data, .4f);
        NewtonRings.Apply(source, new() { NewtonEnabled = true, NewtonAutomatic = true, NewtonField = field, NewtonStrength = 1, NewtonLuminance = 1 });
        Assert.InRange(source.Data[0], .4 * Math.Exp(-.15) - .0001, .4 * Math.Exp(-.15) + .0001);
        Assert.InRange(source.Data[1], .4 * Math.Exp(.06) - .0001, .4 * Math.Exp(.06) + .0001);
    }

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    public void Nonfinite_predictions_never_reach_the_image(float value)
    {
        var residual = new ImageBuffer(1, 1); residual.Data[0] = value;
        Assert.Throws<InvalidDataException>(() => NewtonField.FromLogResidual(residual, new[] { 1f }));
        residual.Data[0] = 0;
        Assert.Throws<InvalidDataException>(() => NewtonField.FromLogResidual(residual, new[] { value }));
    }

    [Fact]
    public void Wrong_confidence_size_and_missing_model_fail_before_applying_a_field()
    {
        Assert.Throws<ArgumentException>(() => NewtonField.FromLogResidual(new ImageBuffer(2, 2), new[] { 1f }));
        Assert.Throws<FileNotFoundException>(() => new OnnxNewtonEstimator(Path.Combine(TestDataIsolation.Root, Guid.NewGuid() + ".onnx")));
    }
}
