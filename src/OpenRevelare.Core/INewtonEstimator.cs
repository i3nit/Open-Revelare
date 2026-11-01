namespace OpenRevelare.Core;

/// <summary>Produces a source-grid log correction; implementations never mutate the input.</summary>
public interface INewtonEstimator : IDisposable
{
    NewtonField Estimate(ImageBuffer source, double sensitivity, CancellationToken cancellation = default);
}
