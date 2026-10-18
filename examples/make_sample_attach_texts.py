"""Regenerates examples/sample_attach_texts.txt, the synthetic corpus the README's attach
walkthrough uses: one text per line, 1,000 lines, deterministic under the fixed seed.

Every line is template-generated here, so nothing in the file comes from any real person,
customer or dataset. It exists so `vegaml.attach` can be tried before you collect your own
domain text, which is what you should actually attach with: the port tunes to the corpus it
is fitted on, and this one is deliberately generic.

    python examples/make_sample_attach_texts.py
"""
import os
import random

PRODUCTS = ["the router", "the dishwasher", "our billing portal", "the mobile app",
            "the thermostat", "the subscription", "the coffee machine", "my laptop",
            "the booking system", "the wireless headphones", "the delivery service",
            "the printer", "our shared calendar", "the smart lock", "the invoice tool"]
PEOPLE = ["my manager", "the landlord", "a colleague", "the support agent", "our supplier",
          "the new tenant", "my neighbour", "the contractor", "the finance team",
          "the delivery driver"]
PLACES = ["the downtown office", "warehouse B", "the Lisbon branch", "the loading dock",
          "meeting room 4", "the staging environment", "the north plant", "the storefront"]
TIMES = ["this morning", "last Tuesday", "over the weekend", "for the third time this month",
         "since the last update", "right before the demo", "during the night shift",
         "an hour ago", "all week"]
AMOUNTS = ["49.99", "212.40", "1,180.00", "16.75", "320.00", "7.20", "89.00", "2,450.00"]
ERRORS = ["ERR_TIMEOUT", "code 502", "a blank screen", "error E-114", "a spinning wheel",
          "an authentication loop", "checksum mismatch", "exit status 3"]
TOPICS = ["the quarterly forecast", "the onboarding checklist", "the migration plan",
          "the vendor contract", "the release schedule", "the training budget",
          "the audit findings", "the roadmap review"]

TEMPLATES = [
    "I was charged {amt} twice for {prod} {time}. Please refund the duplicate and confirm "
    "by email, or I will have to cancel the account.",
    "{prod} stopped working {time} and now shows {err}. I already restarted it twice and "
    "checked the cable. What else can I try before you send a replacement?",
    "Quick note: {pers} asked to move the discussion about {topic} to Thursday. No action "
    "needed from your side yet, just keep the slot free.",
    "Review: {prod} is honestly better than I expected. Setup took five minutes, the manual "
    "was clear, and it has run quietly {time}. Four stars only because the app is slow.",
    "Review: disappointed with {prod}. It looked fine out of the box but failed with {err} "
    "on the second day, and support has not replied {time}.",
    "Incident report, {place}: at 06:40 the sensor on line 2 reported {err}. The shift lead "
    "paused the belt, cleared the jam, and production resumed at 07:05. No injuries.",
    "Reminder that the invoice for {amt} from {pers} is due Friday. Approve it in the portal "
    "or tell me to push it to next month.",
    "Meeting notes: we agreed {pers} owns {topic}, the draft is due in two weeks, and we "
    "revisit the budget once the numbers from {place} arrive.",
    "Can someone confirm whether {prod} supports exporting to CSV? The docs mention it but "
    "the menu at {place} only shows PDF, and the client is asking {time}.",
    "Heads up: {place} reported a water leak {time}. Facilities is on it; keep the server "
    "racks on the east wall powered down until they give the all-clear.",
    "Order update: your package with {prod} left {place} {time} and should arrive within "
    "three working days. Track it with the code in your confirmation email.",
    "I want to praise {pers} for how they handled the outage {time}. Clear updates every "
    "twenty minutes and a full writeup the next day. That is how it should be done.",
    "The recipe calls for two eggs, a cup of flour, and whatever soft fruit you have. Fold "
    "gently, rest the batter {time}, and do not overheat the pan or it burns fast.",
    "Match recap: the home side went ahead early, lost the thread after the break, and only "
    "a save in stoppage time kept the score at 2-1. {pers} will be relieved.",
    "Travel note: the 09:15 connection from {place} is cancelled {time}. Rebooking is free "
    "until midnight; after that a change fee of {amt} applies.",
    "Policy question: does the new expense rule cover meals during travel booked through "
    "{prod}? {pers} says yes, the handbook says ask finance, and finance has not answered.",
    "Log excerpt: job 4821 retried three times {time}, each attempt ending in {err}. The "
    "queue is draining normally otherwise. Suggest raising the timeout before we page anyone.",
    "Please schedule a technician for {prod} at {place}. It powers on but {err} appears "
    "after a minute, and the warranty runs out at the end of the month.",
    "Thanks for the quick turnaround on {topic}. One correction: the figure of {amt} on "
    "page two is last year's; this year's is in the shared sheet, tab three.",
    "Subscription question: if I downgrade {prod} mid-cycle, is the difference of {amt} "
    "credited or lost? The FAQ contradicts what {pers} told me {time}.",
]

LONG_OPENERS = [
    "Full incident timeline, for the record.",
    "Background for whoever picks this ticket up next.",
    "Summary of the week, in some detail because a lot happened.",
    "Complete history of this dispute, since I keep being asked to repeat it.",
]


def make_text(rng):
    t = rng.choice(TEMPLATES)
    return t.format(prod=rng.choice(PRODUCTS), pers=rng.choice(PEOPLE),
                    place=rng.choice(PLACES), time=rng.choice(TIMES),
                    amt=rng.choice(AMOUNTS), err=rng.choice(ERRORS),
                    topic=rng.choice(TOPICS))


def make_long(rng):
    # genuinely long records (several thousand tokens), so an attach over this corpus also
    # exercises the shared-prefix read and the memory-chunk path that long documents take
    parts = [rng.choice(LONG_OPENERS)]
    for _ in range(rng.randint(60, 120)):
        parts.append(make_text(rng))
    return " ".join(parts)


def main():
    rng = random.Random(13)
    n_long = 20
    texts = [make_text(rng) for _ in range(1000 - n_long)] + [make_long(rng) for _ in range(n_long)]
    rng.shuffle(texts)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_attach_texts.txt")
    with open(out, "w") as f:
        f.write("\n".join(texts) + "\n")
    print("wrote %s (%d texts, %d of them long)" % (out, len(texts), n_long))


if __name__ == "__main__":
    main()
