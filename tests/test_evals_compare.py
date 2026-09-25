"""Offline tests for the sweep comparison tool."""

import json

import pytest

from evals import compare


def item(turn, kind, **fields):
    return {"turn": turn, "kind": kind, **fields}


def run(scenario, index, status, items=(), failed=(), warnings=()):
    return {
        "scenario": scenario,
        "run": index,
        "status": status,
        "items": list(items),
        "checks": [{"name": n, "passed": False, "detail": ""} for n in failed]
        + [{"name": "fine", "passed": True, "detail": ""}],
        "warnings": [{"name": n, "fired": True, "detail": ""} for n in warnings],
    }


def test_replay_finds_what_the_guard_would_have_blocked_in_old_transcripts():
    runs = [
        run(
            "a",
            0,
            "fail",
            [
                item(0, "say", text="Здравствуйте! Хорошо, записал."),  # the greeting is code-made
                item(1, "say", text="Хорошо, записал."),
                item(1, "say", text="Номер — 8 916 123 45 67."),
                item(2, "say", text="Заявка принята, администратор перезвонит."),
            ],
        ),
    ]

    assert compare.replay_guard(runs) == {
        "written_down": 1,
        "phone_digits": 1,
        "acceptance_claim": 1,
    }


def test_replay_skips_sentences_the_code_wrote_and_respects_saves():
    runs = [
        run(
            "a",
            0,
            "pass",
            [
                item(1, "tool", tool="prepare_booking", text="ok"),
                item(1, "say", text="Проверьте: номер заканчивается на четыре пять шесть семь."),
                item(
                    1, "say", text="Ваш номер 8 916 123 45 67. Записал."
                ),  # code-made, after the tool
                item(2, "tool", tool="confirm_booking", committed=True, text="ok"),
                item(
                    2, "say", text="Заявка принята и передана администратору."
                ),  # fine after a save
                item(3, "say", text="Записал вас."),  # fine after a save
            ],
        ),
    ]

    assert compare.replay_guard(runs) == {}


def test_actual_blocks_are_used_when_the_sweep_had_the_guard_on():
    runs = [
        run("a", 0, "pass", [item(1, "blocked", rule="written_down", text="x")]),
        run(
            "a",
            1,
            "pass",
            [
                item(1, "blocked", rule="written_down", text="y"),
                item(1, "blocked", rule="phone_digits", text="z"),
            ],
        ),
    ]

    counts, how = compare.guard_blocks(runs)

    assert how == "actual" and counts == {"written_down": 2, "phone_digits": 1}


def test_a_sweep_without_the_guard_is_replayed():
    counts, how = compare.guard_blocks(
        [run("a", 0, "pass", [item(1, "say", text="Хорошо, записал.")])]
    )
    assert how == "replayed" and counts == {"written_down": 1}


def test_failure_kinds_count_runs_and_skip_infra_and_inconclusive():
    runs = [
        run("a", 0, "fail", failed=["phone_ok", "date_ok"]),
        run("a", 1, "fail", failed=["phone_ok"]),
        run("a", 2, "infra_error", failed=["phone_ok"]),
        run("a", 3, "pass"),
    ]
    assert compare.failure_kinds(runs) == {"phone_ok": 2, "date_ok": 1}


def test_summary_and_per_scenario_rates():
    runs = [
        run("a", 0, "pass"),
        run("a", 1, "fail", failed=["x"]),
        run("b", 0, "pass"),
        run("b", 1, "infra_error"),
        run("b", 2, "inconclusive"),
    ]

    summary = compare.summarize(runs)

    assert (summary["runs"], summary["gradable"], summary["passed"]) == (5, 3, 2)
    assert (summary["infra"], summary["inconclusive"]) == (1, 1)
    assert summary["per_scenario"] == {"a": [1, 2], "b": [1, 1]}


def test_the_comparison_shows_counts_per_100_runs_and_how_guard_numbers_were_obtained():
    ref = compare.summarize(
        [
            run(
                "a",
                i,
                "fail" if i < 5 else "pass",
                [item(1, "say", text="Хорошо, записал.")],
                failed=["no_full_phone_spoken"] if i < 5 else [],
                warnings=["own_recap_before_prepare"],
            )
            for i in range(10)
        ]
    )
    new = compare.summarize(
        [
            run(
                "a",
                i,
                "pass",
                [item(1, "blocked", rule="written_down", text="x")] if i == 0 else [],
            )
            for i in range(20)
        ]
    )

    text = compare.format_comparison(ref, new, "reference", "new")

    assert "raw pass rate" in text and "5/10 = 50%" in text and "20/20 = 100%" in text
    assert "no_full_phone_spoken" in text and "5 (  50/100)" in text
    assert "reference: replayed, new: actual" in text
    assert "written_down" in text and "own_recap_before_prepare" in text


def test_the_command_line_reads_two_result_directories(tmp_path, capsys):
    for name, status in (("ref", "fail"), ("new", "pass")):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "results.json").write_text(
            json.dumps([run("a", 0, status, failed=["x"] if status == "fail" else [])]),
            encoding="utf-8",
        )

    assert compare.main([str(tmp_path / "ref"), str(tmp_path / "new")]) == 0
    assert "0/1 = 0%" in capsys.readouterr().out


def test_a_missing_results_file_is_reported(tmp_path, capsys):
    assert compare.main([str(tmp_path), str(tmp_path)]) == 1
    assert "cannot read the results" in capsys.readouterr().err


@pytest.mark.parametrize("kind", ["say", "tool"])
def test_old_result_files_without_turn_zero_are_tolerated(kind):
    """The first sweeps' JSON export dropped turn=0 (0 == False): items may lack 'turn'."""
    old = [{"kind": kind, "text": "Хорошо, записал.", "tool": "x"}]
    assert compare.replay_guard([run("a", 0, "pass", old)]) == {}  # turn 0 = greeting: skipped
