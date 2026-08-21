#!/usr/bin/env python3
"""Generate original, safety-cue-free legitimate service-call openings."""

# ruff: noqa: E501 -- full dialogue turns remain intact for label review.

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

GENERATOR_VERSION = 1
DEFAULT_ROWS_PER_SCENARIO = 64
STRUCTURES = ("inbound", "callback", "transfer", "outbound_update")
SOURCE = "scamguard_synthetic_legitimate_call_openings_v1"
SYNTHETIC_METHOD = "deterministic_service_dialogue_error_audit_grounded_original_copy"
PATTERN_REFERENCE = "https://consumer.ftc.gov/articles/how-avoid-scam"
FORBIDDEN_SAFETY_CUES = (
    "never ask",
    "will not ask",
    "do not share",
    "official app",
    "gift card",
    "security code",
    "this is not a scam",
)

SCENARIOS: dict[str, dict[str, tuple[str, ...]]] = {
    "account_service": {
        "desk": ("account service desk", "member services", "card support team", "branch support"),
        "topic": ("a card replacement", "a statement question", "a recurring charge", "an account preference"),
        "request": ("I would like to review when the replacement should arrive", "I need help finding the statement entry", "I want to understand how the recurring charge is listed", "I am trying to update the preference I selected earlier"),
        "detail": ("The note is already attached to the service request", "I have the account page open while we talk", "The branch created the request after my visit", "I received the routine service notice this morning"),
        "action": ("I can compare the available delivery windows", "I can walk through the statement description", "I can check how the merchant entry was categorized", "I can read back the preference currently on file"),
    },
    "parcel_service": {
        "desk": ("parcel service desk", "delivery support", "local dispatch team", "collection-point service"),
        "topic": ("a delivery window", "a collection request", "a damaged parcel", "a delivery instruction"),
        "request": ("I need to move the delivery to a day when someone is home", "I would like the parcel held at the staffed collection point", "I want to add a note about the damaged outer box", "I need to clarify where the driver should leave the package"),
        "detail": ("The tracking page shows it at the local depot", "The sender said the collection option is available", "I took note of the condition when it arrived", "The building desk accepts deliveries during the afternoon"),
        "action": ("I can review the routes serving your area", "I can check the collection-point schedule", "I can open a packaging review for the depot", "I can add a driver note to the existing shipment"),
    },
    "insurance_service": {
        "desk": ("policy service team", "claims support", "member coverage desk", "repair coordination team"),
        "topic": ("a repair estimate", "a policy update", "a claim appointment", "a coverage question"),
        "request": ("I am calling to compare the repair appointment options", "I need to update how the property is described", "I would like to move the claim inspection", "I want to understand which service category applies"),
        "detail": ("The adjuster added notes after the initial visit", "The renewal summary prompted my question", "The repair shop has offered two possible dates", "The member portal shows the request as pending review"),
        "action": ("I can read the appointment choices from the claim", "I can document the updated property description", "I can coordinate a different inspection window", "I can explain the category shown in the coverage summary"),
    },
    "health_scheduling": {
        "desk": ("clinic scheduling", "patient appointment desk", "therapy scheduling team", "community health reception"),
        "topic": ("a follow-up visit", "a therapy appointment", "a routine consultation", "a clinic location change"),
        "request": ("I need a later time for the follow-up visit", "I am trying to schedule the next therapy session", "I would like to move the routine consultation", "I want to check whether the other clinic has availability"),
        "detail": ("The clinician suggested arranging it after my workday", "The reception note says the next session is ready to book", "My calendar changed after the original appointment was made", "The alternate clinic is easier for me to reach"),
        "action": ("I can compare the open appointment windows", "I can check the therapist's next clinic day", "I can move the consultation to another afternoon", "I can look at both clinic schedules together"),
    },
    "travel_service": {
        "desk": ("travel service desk", "rail booking support", "holiday reservations", "journey planning team"),
        "topic": ("a seat preference", "a return journey", "a hotel change", "a connection question"),
        "request": ("I would like to change the seat preference on my journey", "I need a later return journey on the same day", "I am calling to compare two hotel dates", "I want to allow more time for the connection"),
        "detail": ("The booking summary is in front of me", "The event schedule changed after I booked", "The rest of the itinerary can stay as it is", "The current connection feels a little too short"),
        "action": ("I can review the seats still available", "I can compare the later return services", "I can check availability around both dates", "I can find an itinerary with a longer connection"),
    },
    "telecom_service": {
        "desk": ("broadband support", "mobile service team", "home phone desk", "network appointment team"),
        "topic": ("an installation visit", "a signal question", "a plan change", "a router delivery"),
        "request": ("I need to move the installation visit to the afternoon", "I am calling about a signal problem in one room", "I would like to compare the current plan with the smaller option", "I want to check when the replacement router should arrive"),
        "detail": ("The engineer left an appointment note yesterday", "The service works normally in the rest of the home", "My usage has changed since the plan was selected", "The dispatch notice says the parcel has left the warehouse"),
        "action": ("I can check the engineer's afternoon route", "I can add the room detail to the support case", "I can explain the differences between the two plans", "I can review the delivery estimate with you"),
    },
    "energy_service": {
        "desk": ("home energy service", "meter appointment desk", "utility customer team", "energy efficiency bookings"),
        "topic": ("a meter visit", "a billing period", "a home assessment", "a service address note"),
        "request": ("I need to change the time of the meter visit", "I have a question about which dates the bill covers", "I would like to arrange the home assessment", "I want to add an access note for the engineer"),
        "detail": ("The appointment letter arrived earlier this week", "The billing period crosses the date I moved", "The adviser suggested a morning assessment", "The side entrance is easier for service visits"),
        "action": ("I can compare the remaining visit windows", "I can explain the dates shown on the account", "I can book the next assessment slot", "I can attach the access note to the work order"),
    },
    "retail_service": {
        "desk": ("store customer care", "order support", "returns desk", "product service team"),
        "topic": ("an exchange", "a collection order", "a missing accessory", "a warranty appointment"),
        "request": ("I would like a different size in the same style", "I need to move the collection to another store", "I am calling because one accessory was not in the box", "I want to arrange a time for the product inspection"),
        "detail": ("The purchase appears in my store account", "The order is still waiting for collection", "The packing list shows the accessory", "The service desk asked me to book before visiting"),
        "action": ("I can check that size at nearby stores", "I can compare the collection locations", "I can record the missing item against the order", "I can view the next inspection appointments"),
    },
    "hospitality_service": {
        "desk": ("guest reservations", "hotel reception", "conference services", "accommodation team"),
        "topic": ("an arrival time", "a room preference", "a meeting-room booking", "a breakfast request"),
        "request": ("I would like to note that I will arrive later in the evening", "I am calling to ask about a quieter room", "I need to adjust the meeting-room start time", "I want to add breakfast to the second morning"),
        "detail": ("The reservation confirmation has the earlier time", "The current room preference is flexible", "The attendee schedule moved slightly", "The other parts of the reservation are correct"),
        "action": ("I can add the later arrival note", "I can compare the rooms available for those dates", "I can check the room schedule around your meeting", "I can update the breakfast count for that morning"),
    },
    "food_service": {
        "desk": ("catering service", "grocery delivery help", "restaurant bookings", "meal order team"),
        "topic": ("a menu choice", "a delivery substitution", "a table booking", "a collection time"),
        "request": ("I need to change one menu choice for the group", "I would like a different substitution for the unavailable item", "I am calling to add another guest to the table", "I want to move the collection to later in the evening"),
        "detail": ("The event coordinator asked me to update the meal count", "The order page lists two possible alternatives", "The booking was made for a smaller group", "The kitchen said a later collection may be possible"),
        "action": ("I can update the group menu note", "I can record which alternative you prefer", "I can check whether a larger table is open", "I can compare the later collection times"),
    },
    "property_service": {
        "desk": ("property viewing team", "tenant service desk", "building maintenance", "lettings support"),
        "topic": ("a viewing time", "a maintenance visit", "a move-in question", "a building access request"),
        "request": ("I would like to see the property later in the day", "I need to move the maintenance visit", "I have a question about the move-in schedule", "I want to add a note for the building caretaker"),
        "detail": ("The viewing invitation lists a morning time", "The repair request is already with the contractor", "The welcome information mentions two possible dates", "The front entrance is closed during the planned visit"),
        "action": ("I can compare the later viewing times", "I can look at the contractor's next route", "I can explain the sequence shown in the welcome pack", "I can add the caretaker note to the request"),
    },
    "aviation_service": {
        "desk": ("airline service desk", "airport assistance", "flight reservations", "baggage support"),
        "topic": ("a seat assignment", "airport assistance", "a flight change", "a delayed bag"),
        "request": ("I would like to sit closer to the front of the cabin", "I need to arrange assistance between the gate and the aircraft", "I am calling to compare two flights on the same day", "I want to add a delivery preference for the delayed bag"),
        "detail": ("The trip summary shows the current seat", "The connection uses a different terminal", "My meeting ends earlier than expected", "The baggage desk opened the case at the airport"),
        "action": ("I can review the open seats", "I can add the assistance request to the journey", "I can compare the two flight schedules", "I can update the delivery preference on the baggage case"),
    },
    "equipment_service": {
        "desk": ("equipment parts desk", "field service bookings", "machinery support", "cooperative service team"),
        "topic": ("a replacement part", "a service visit", "an equipment manual", "a seasonal maintenance slot"),
        "request": ("I am trying to confirm which replacement part matches the attachment", "I need a different day for the field service visit", "I would like the manual for the older model", "I want to arrange maintenance before the busy season"),
        "detail": ("The workshop wrote the model description on the service note", "The machine is available later in the week", "The cover plate has the model family printed on it", "The cooperative suggested booking ahead this year"),
        "action": ("I can compare the attachment descriptions", "I can check the technician's later route", "I can locate the matching manual edition", "I can review the seasonal service calendar"),
    },
    "technology_service": {
        "desk": ("device support", "software customer care", "warranty service", "workplace technology desk"),
        "topic": ("a warranty repair", "a subscription setting", "a printer setup", "a software appointment"),
        "request": ("I want to arrange the warranty inspection", "I need help locating the subscription preference", "I am calling about the printer setup at my desk", "I would like to move the software appointment"),
        "detail": ("The service page shows the repair request as accepted", "The setting is not where it appeared in the previous version", "The printer is visible but the queue stays paused", "The calendar invitation no longer fits my schedule"),
        "action": ("I can compare the inspection locations", "I can describe where the preference now appears", "I can work through the local queue settings", "I can offer the next appointment windows"),
    },
    "financial_planning": {
        "desk": ("planning appointment team", "pension service desk", "savings review team", "financial guidance bookings"),
        "topic": ("a review meeting", "a pension statement", "a savings appointment", "a document question"),
        "request": ("I would like a later time for the review meeting", "I have a question about the dates on the pension statement", "I need to move the savings appointment", "I want to know which document the meeting note refers to"),
        "detail": ("The adviser sent the meeting summary after our last discussion", "The latest statement covers part of the previous year", "My work schedule changed this week", "The appointment checklist uses a short document title"),
        "action": ("I can compare the adviser's later openings", "I can explain the statement period shown", "I can move the appointment to another day", "I can expand the document title from the checklist"),
    },
    "entertainment_service": {
        "desk": ("venue box office", "membership services", "event customer care", "theatre bookings"),
        "topic": ("a seat exchange", "a membership renewal", "an event time", "an accessibility request"),
        "request": ("I would like to exchange the seats for the afternoon performance", "I am calling to change the start month of the membership", "I need to check which event time is on my booking", "I want to add an aisle-seat preference"),
        "detail": ("The booking confirmation lists the evening performance", "The current membership runs a little longer than I expected", "The venue calendar shows two performances that day", "The access team said the preference can be added in advance"),
        "action": ("I can compare the afternoon seating sections", "I can review the renewal start choices", "I can match the booking with the venue calendar", "I can add the aisle preference to the booking"),
    },
}


def short_hash(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:length]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render(spec: dict[str, tuple[str, ...]], structure: str, variant: int) -> str:
    desk = spec["desk"][(variant + len(structure)) % 4]
    topic = spec["topic"][(variant * 3 + len(structure)) % 4]
    request = spec["request"][variant % 4]
    detail = spec["detail"][(variant // 4) % 4]
    action = spec["action"][(variant * 3 + variant // 4) % 4]
    if structure == "inbound":
        return (
            f"AGENT: Good morning, you've reached the {desk}. What can I help with?\n"
            f"CUSTOMER: {request}.\n"
            f"AGENT: Certainly. {action}.\n"
            f"CUSTOMER: Thanks. {detail}."
        )
    if structure == "callback":
        return (
            f"AGENT: Hello, I'm returning the callback requested about {topic}. Is now still a good time?\n"
            f"CUSTOMER: Yes, thank you. {detail}.\n"
            f"AGENT: Great. {action}.\n"
            f"CUSTOMER: That works; {request.casefold()}."
        )
    if structure == "transfer":
        return (
            f"CUSTOMER: Hi, I was transferred to the {desk} about {topic}.\n"
            f"AGENT: Yes, I have the service note here. How would you like to proceed?\n"
            f"CUSTOMER: {request}. {detail}.\n"
            f"AGENT: All right. {action}."
        )
    if structure == "outbound_update":
        return (
            f"AGENT: Hi, this is the {desk} with the update you requested about {topic}.\n"
            f"CUSTOMER: Thanks for getting back to me. {detail}.\n"
            f"AGENT: I see that note. {action}.\n"
            f"CUSTOMER: Good, {request.casefold()}."
        )
    raise ValueError(f"unknown dialogue structure: {structure}")


def generate(rows_per_scenario: int = DEFAULT_ROWS_PER_SCENARIO) -> list[dict[str, object]]:
    if rows_per_scenario < len(STRUCTURES) or rows_per_scenario % len(STRUCTURES):
        raise ValueError("rows per scenario must be a positive multiple of four")
    rows: list[dict[str, object]] = []
    for scenario, spec in SCENARIOS.items():
        for index in range(rows_per_scenario):
            structure = STRUCTURES[index % len(STRUCTURES)]
            variant = index // len(STRUCTURES)
            text = render(spec, structure, variant)
            folded = text.casefold()
            if any(cue in folded for cue in FORBIDDEN_SAFETY_CUES):
                raise ValueError(f"explicit safety cue leaked into {scenario}:{index}")
            row_id = "legit-call-" + short_hash(
                f"v{GENERATOR_VERSION}:{scenario}:{structure}:{variant}:{text}"
            )
            rows.append(
                {
                    "id": row_id,
                    "text": text,
                    "label": "SAFE",
                    "category": "NONE",
                    "source": SOURCE,
                    "source_label": "synthetic_legitimate_service_call",
                    "license": "Apache-2.0",
                    "split": "train",
                    "family_id": (
                        f"synthetic:legitimate_call_opening:{scenario}:{structure}:"
                        f"v{GENERATOR_VERSION}"
                    ),
                    "is_synthetic": True,
                    "synthetic_method": SYNTHETIC_METHOD,
                    "pattern_reference": PATTERN_REFERENCE,
                    "source_language": "English",
                    "scenario": scenario,
                    "dialogue_structure": structure,
                    "generator_version": GENERATOR_VERSION,
                    "selection_signal": (
                        "open AppTek selection localized false positives to early call windows"
                    ),
                    "external_benchmark_text_copied": False,
                    "context_policy": "four_turn_service_opening_under_mobile_256_token_window",
                }
            )
    texts = {" ".join(str(row["text"]).casefold().split()) for row in rows}
    ids = {str(row["id"]) for row in rows}
    if len(texts) != len(rows) or len(ids) != len(rows):
        raise ValueError("legitimate-call generator produced a duplicate row")
    return sorted(rows, key=lambda row: str(row["id"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-per-scenario", type=int, default=DEFAULT_ROWS_PER_SCENARIO)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/generated/legitimate_call_openings.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/generated/legitimate_call_openings_manifest.json"),
    )
    args = parser.parse_args()
    rows = generate(args.rows_per_scenario)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "source": SOURCE,
        "license": "Apache-2.0",
        "method": SYNTHETIC_METHOD,
        "used_for_fitting": True,
        "used_for_threshold": False,
        "external_benchmark_text_copied": False,
        "design_signal": "open AppTek selection metrics and metadata slices only",
        "rows": len(rows),
        "rows_per_scenario": args.rows_per_scenario,
        "labels": dict(Counter(str(row["label"]) for row in rows)),
        "scenarios": dict(Counter(str(row["scenario"]) for row in rows)),
        "dialogue_structures": dict(
            Counter(str(row["dialogue_structure"]) for row in rows)
        ),
        "families": len({str(row["family_id"]) for row in rows}),
        "sha256": file_sha256(args.output),
        "pattern_reference": PATTERN_REFERENCE,
        "forbidden_safety_cues": list(FORBIDDEN_SAFETY_CUES),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
