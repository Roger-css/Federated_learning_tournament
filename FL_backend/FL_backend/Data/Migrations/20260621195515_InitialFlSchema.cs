using System;
using Microsoft.EntityFrameworkCore.Migrations;
using Npgsql.EntityFrameworkCore.PostgreSQL.Metadata;

#nullable disable

namespace FL_backend.Data.Migrations
{
    /// <inheritdoc />
    public partial class InitialFlSchema : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.EnsureSchema(
                name: "fl");

            migrationBuilder.CreateTable(
                name: "fl_local_baseline",
                schema: "fl",
                columns: table => new
                {
                    Id = table.Column<int>(type: "integer", nullable: false)
                        .Annotation("Npgsql:ValueGenerationStrategy", NpgsqlValueGenerationStrategy.IdentityByDefaultColumn),
                    ClientId = table.Column<string>(type: "character varying(100)", maxLength: 100, nullable: false),
                    TrainF1 = table.Column<double>(type: "double precision", nullable: false),
                    TestF1 = table.Column<double>(type: "double precision", nullable: false),
                    Accuracy = table.Column<double>(type: "double precision", nullable: false),
                    NumExamples = table.Column<int>(type: "integer", nullable: false),
                    RecordedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false, defaultValueSql: "NOW()")
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_fl_local_baseline", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "fl_round",
                schema: "fl",
                columns: table => new
                {
                    Id = table.Column<int>(type: "integer", nullable: false)
                        .Annotation("Npgsql:ValueGenerationStrategy", NpgsqlValueGenerationStrategy.IdentityByDefaultColumn),
                    RoundNumber = table.Column<int>(type: "integer", nullable: false),
                    RecordedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false, defaultValueSql: "NOW()")
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_fl_round", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "fl_client_result",
                schema: "fl",
                columns: table => new
                {
                    Id = table.Column<int>(type: "integer", nullable: false)
                        .Annotation("Npgsql:ValueGenerationStrategy", NpgsqlValueGenerationStrategy.IdentityByDefaultColumn),
                    FlRoundId = table.Column<int>(type: "integer", nullable: false),
                    ClientId = table.Column<string>(type: "character varying(100)", maxLength: 100, nullable: false),
                    TrainF1 = table.Column<double>(type: "double precision", nullable: false),
                    TestF1 = table.Column<double>(type: "double precision", nullable: false),
                    Accuracy = table.Column<double>(type: "double precision", nullable: false),
                    NumExamples = table.Column<int>(type: "integer", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_fl_client_result", x => x.Id);
                    table.ForeignKey(
                        name: "FK_fl_client_result_fl_round_FlRoundId",
                        column: x => x.FlRoundId,
                        principalSchema: "fl",
                        principalTable: "fl_round",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateIndex(
                name: "IX_fl_client_result_FlRoundId",
                schema: "fl",
                table: "fl_client_result",
                column: "FlRoundId");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "fl_client_result",
                schema: "fl");

            migrationBuilder.DropTable(
                name: "fl_local_baseline",
                schema: "fl");

            migrationBuilder.DropTable(
                name: "fl_round",
                schema: "fl");
        }
    }
}
