namespace FL_backend.Models;

public class FlLocalBaseline
{
    public int Id { get; set; }
    public string ClientId { get; set; } = string.Empty;
    public double TrainF1 { get; set; }
    public double TestF1 { get; set; }
    public double Accuracy { get; set; }
    public int NumExamples { get; set; }
    public DateTime RecordedAt { get; set; } = DateTime.UtcNow;
}
