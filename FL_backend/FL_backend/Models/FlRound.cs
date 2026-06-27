namespace FL_backend.Models;

public class FlRound
{
    public int Id { get; set; }
    public int RoundNumber { get; set; }
    public DateTime RecordedAt { get; set; } = DateTime.UtcNow;
    public List<FlClientResult> ClientResults { get; set; } = new();
}
