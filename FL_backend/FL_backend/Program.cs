using FL_backend.Data;
using Microsoft.EntityFrameworkCore;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var connString = builder.Configuration.GetConnectionString("FL_Context")
    ?? throw new InvalidOperationException("Connection string 'FL_Context' not found.");
builder.Services.AddDbContext<FLContext>(options =>
    options.UseNpgsql(connString));

var app = builder.Build();

app.Logger.LogInformation("Using connection string: {ConnString}", connString);

// Apply EF migrations automatically on startup (dev only)
using (var scope = app.Services.CreateScope())
{
    var db = scope.ServiceProvider.GetRequiredService<FLContext>();
    try
    {
        await db.Database.EnsureCreatedAsync();
        await db.Database.MigrateAsync();
        app.Logger.LogInformation("EF migrations applied successfully.");
    }
    catch (Exception ex)
    {
        app.Logger.LogWarning(ex, "Could not apply EF migrations; database may not be ready.");
    }
}

if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseAuthorization();
app.MapControllers();

app.Run();
