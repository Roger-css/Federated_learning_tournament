using FL_backend.Data;
using FL_backend.Hubs;
using Microsoft.EntityFrameworkCore;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();
builder.Services.AddSignalR();
builder.Services.AddCors(options =>
{
    options.AddPolicy("FrontendPolicy", policy =>
        policy.WithOrigins("http://localhost:3000")
              .AllowAnyHeader()
              .AllowAnyMethod()
              .AllowCredentials());
});

var connString = builder.Configuration.GetConnectionString("FL_Context")
    ?? throw new InvalidOperationException("Connection string 'FL_Context' not found.");
builder.Services.AddDbContext<FLContext>(options =>
    options.UseNpgsql(connString));

var app = builder.Build();

app.Logger.LogInformation("Using connection string: {ConnString}", connString);

// Ensure database schema exists (idempotent — no-op if already created)
using (var scope = app.Services.CreateScope())
{
    var db = scope.ServiceProvider.GetRequiredService<FLContext>();
    await db.Database.EnsureCreatedAsync();
}

if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseCors("FrontendPolicy");
app.UseAuthorization();
app.MapControllers();
app.MapHub<FlHub>("/hubs/fl");

app.Run();
