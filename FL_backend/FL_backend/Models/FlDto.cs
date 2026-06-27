using System.Text.Json.Serialization;

namespace FL_backend.Models;

public class LocalBaselineRequest
{
    [JsonPropertyName("clients")]
    public List<ClientMetric> Clients { get; set; } = new();
}

public class RoundRequest
{
    [JsonPropertyName("roundNumber")]
    public int RoundNumber { get; set; }

    [JsonPropertyName("clients")]
    public List<ClientMetric> Clients { get; set; } = new();
}

public class ClientMetric
{
    [JsonPropertyName("clientId")]
    public string ClientId { get; set; } = string.Empty;

    [JsonPropertyName("trainF1")]
    public double TrainF1 { get; set; }

    [JsonPropertyName("testF1")]
    public double TestF1 { get; set; }

    [JsonPropertyName("accuracy")]
    public double Accuracy { get; set; }

    [JsonPropertyName("numExamples")]
    public int NumExamples { get; set; }
}
