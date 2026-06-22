using FL_backend.Data;
using FL_backend.Hubs;
using FL_backend.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.SignalR;
using Microsoft.EntityFrameworkCore;

namespace FL_backend.Controllers;

[ApiController]
[Route("api/fl")]
public class FlController : ControllerBase
{
    private readonly FLContext _db;
    private readonly ILogger<FlController> _logger;
    private readonly IHubContext<FlHub> _hubContext;

    public FlController(FLContext db, ILogger<FlController> logger, IHubContext<FlHub> hubContext)
    {
        _db = db;
        _logger = logger;
        _hubContext = hubContext;
    }

    [HttpGet("local-baseline")]
    public async Task<IActionResult> GetLocalBaseline()
    {
        var data = await _db.FlLocalBaselines
            .OrderByDescending(r => r.RecordedAt)
            .Select(r => new
            {
                clientId = r.ClientId,
                trainF1 = r.TrainF1,
                testF1 = r.TestF1,
                accuracy = r.Accuracy,
                numExamples = r.NumExamples,
                recordedAt = r.RecordedAt,
            })
            .ToListAsync();

        return Ok(new { clients = data });
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

            var savedClients = rows.Select(r => new
            {
                clientId = r.ClientId,
                trainF1 = r.TrainF1,
                testF1 = r.TestF1,
                accuracy = r.Accuracy,
                numExamples = r.NumExamples,
            }).ToList();
            await _hubContext.Clients.All.SendAsync("LocalBaselineUpdated", savedClients);

            return StatusCode(201, new { saved = rows.Count });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to save local baseline.");
            return StatusCode(500, new { error = "An internal error occurred." });
        }
    }

    [HttpGet("rounds")]
    public async Task<IActionResult> GetRounds()
    {
        var data = await _db.FlRounds
            .Include(r => r.ClientResults)
            .OrderByDescending(r => r.RoundNumber)
            .Select(r => new
            {
                roundNumber = r.RoundNumber,
                recordedAt = r.RecordedAt,
                clients = r.ClientResults.Select(c => new
                {
                    clientId = c.ClientId,
                    trainF1 = c.TrainF1,
                    testF1 = c.TestF1,
                    accuracy = c.Accuracy,
                    numExamples = c.NumExamples,
                }).ToList(),
            })
            .ToListAsync();

        return Ok(new { rounds = data });
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

            var savedClientResults = round.ClientResults.Select(c => new
            {
                clientId = c.ClientId,
                trainF1 = c.TrainF1,
                testF1 = c.TestF1,
                accuracy = c.Accuracy,
                numExamples = c.NumExamples,
            }).ToList();
            await _hubContext.Clients.All.SendAsync("RoundUpdated", new
            {
                roundNumber = request.RoundNumber,
                clients = savedClientResults
            });

            return StatusCode(201, new { roundId = round.Id, saved = round.ClientResults.Count });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to save round {RoundNumber}.", request.RoundNumber);
            return StatusCode(500, new { error = "An internal error occurred." });
        }
    }
}
