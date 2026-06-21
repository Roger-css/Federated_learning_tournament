using FL_backend.Models;
using Microsoft.EntityFrameworkCore;

namespace FL_backend.Data;

public class FLContext : DbContext
{
    public FLContext(DbContextOptions<FLContext> options) : base(options) { }

    public DbSet<FlLocalBaseline> FlLocalBaselines => Set<FlLocalBaseline>();
    public DbSet<FlRound> FlRounds => Set<FlRound>();
    public DbSet<FlClientResult> FlClientResults => Set<FlClientResult>();

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        modelBuilder.HasDefaultSchema("fl");

        modelBuilder.Entity<FlLocalBaseline>(entity =>
        {
            entity.ToTable("fl_local_baseline");
            entity.HasKey(e => e.Id);
            entity.Property(e => e.ClientId).IsRequired().HasMaxLength(100);
            entity.Property(e => e.RecordedAt).HasDefaultValueSql("NOW()");
        });

        modelBuilder.Entity<FlRound>(entity =>
        {
            entity.ToTable("fl_round");
            entity.HasKey(e => e.Id);
            entity.Property(e => e.RecordedAt).HasDefaultValueSql("NOW()");
        });

        modelBuilder.Entity<FlClientResult>(entity =>
        {
            entity.ToTable("fl_client_result");
            entity.HasKey(e => e.Id);
            entity.Property(e => e.ClientId).IsRequired().HasMaxLength(100);
            entity.HasOne(e => e.FlRound)
                  .WithMany(r => r.ClientResults)
                  .HasForeignKey(e => e.FlRoundId)
                  .OnDelete(DeleteBehavior.Cascade);
        });

        base.OnModelCreating(modelBuilder);
    }
}
