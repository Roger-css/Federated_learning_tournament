using FL_backend.Data;
using FL_backend.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace FL_backend.Controllers;

[ApiController]
[Route("api/fl")]
public class FlController : ControllerBase
{
    private readonly FLContext _db;
    private readonly ILogger<FlController> _logger;

    public FlController(FLContext db, ILogger<FlController> logger)
    {
        _db = db;
        _logger = logger;
    }

    [HttpPost("local-baseline")]
    public async Task<IActionResult> PostLocalBaseline([FromBody] LocalBaselineRequest request)
    {
        if (request.Clients == null || request.Clients.Count == 0)
            return BadRequest(new { error = "clients list is required and must not be empty." });

        try
        {
            var rows = request.Clients.Select(c => new FlLocalBaseline
            {
                ClientId = c.ClientId,
                TrainF1 = c.TrainF1,
                TestF1 = c.TestF1,
                Accuracy = c.Accuracy,
                NumExamples = c.NumExamples,
            }).ToList();

            _db.FlLocalBaselines.AddRange(rows);
            await _db.SaveChangesAsync();

            _logger.LogInformation("Saved {Count} local baseline rows.", rows.Count);
            return StatusCode(201, new { saved = rows.Count });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to save local baseline.");
            return StatusCode(500, new { error = "An internal error occurred." });
        }
    }

    [HttpPost("rounds")]
    public async Task<IActionResult> PostRound([FromBody] RoundRequest request)
    {
        if (request.Clients == null || request.Clients.Count == 0)
            return BadRequest(new { error = "clients list is required and must not be empty." });

        if (request.RoundNumber <= 0)
            return BadRequest(new { error = "roundNumber must be positive." });

        try
        {
            var round = new FlRound
            {
                RoundNumber = request.RoundNumber,
                ClientResults = request.Clients.Select(c => new FlClientResult
                {
                    ClientId = c.ClientId,
                    TrainF1 = c.TrainF1,
                    TestF1 = c.TestF1,
                    Accuracy = c.Accuracy,
                    NumExamples = c.NumExamples,
                }).ToList(),
            };

            _db.FlRounds.Add(round);
            await _db.SaveChangesAsync();

            _logger.LogInformation("Saved round {RoundNumber} with {Count} client results.",
                round.RoundNumber, round.ClientResults.Count);
            return StatusCode(201, new { roundId = round.Id, saved = round.ClientResults.Count });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to save round {RoundNumber}.", request.RoundNumber);
            return StatusCode(500, new { error = "An internal error occurred." });
        }
    }
}
