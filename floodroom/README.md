# FloodPrint Room 🌊

**A flood watch that lives in your neighbourhood group chat.**

Built for the [Flip the Switch Hackathon](https://www.wemakedevs.org/hackathons/switch) (WeMakeDevs × DevTools Academy × Switch) by team **Neural Builders**.

Every rainy season in Port Harcourt, the same thing happens: someone posts "is Rumuola flooded?" in the estate group, five people answer from memory, nobody is sure, and next week the same question comes again. The information exists. It's in people's eyes and phones. It just disappears into the chat scroll.

FloodPrint Room puts a [Switch](https://github.com/sandbox-quantum/switch) AI teammate **into that group chat**. Residents keep talking the way they already do. The agent listens when addressed, logs each sighting, confirms it when a second person agrees, tells travellers which road to avoid, and **remembers** every flood so the city builds its own flood map, one message at a time.

No app to install. No form. Just the chat.

> Demo video: _add your link here_

```text
Ada:        @floodprint Rumuola junction is knee deep, cars turning back  [📷 photo]
FloodPrint: 📍 FP-0013 logged: Rumuola, knee deep, by Ada (with photo)
            🟠 KNEE: avoid in small cars, walk with care
            👤 1 person so far. One more report from someone else confirms it.
            🧠 The room remembers: Rumuola has flooded on 5 days on record.
            Thanks Ada, you're the sensor today.

Tonye:      @floodprint same here at Rumuola, knee level
FloodPrint: 📍 FP-0014 logged: Rumuola, knee deep, by Tonye
            🟠 KNEE: avoid in small cars, walk with care
            👥 CONFIRMED: 2 reports from 2 people in the last 6h

Chidi:      @floodprint I'm going Rumuokoro to Mile 1 now, which way?
FloodPrint: 📍 Rumuokoro  ⚪ no reports in the last 6h ...
            📍 Rumuola    🟠 KNEE (CONFIRMED) ...
            📍 Mile 1 Diobu ⚪ no reports in the last 6h ...
            Avoid the Rumuola leg; go through Rumuigbo instead.

Ibim:       @floodprint water don comot for Rumuola, e don dry
FloodPrint: ✅ FP-0015: Rumuola marked clear by Ibim. Thank you.
```

## How Switch is used

Switch is the whole delivery layer. Without it this would be one more website nobody opens during a storm.

| Switch feature | What FloodPrint Room does with it |
| --- | --- |
| **Messaging bridge** (Telegram here; Discord, Slack, Teams and Mattermost work the same way) | The agent sits in the residents' existing group. Nobody moves to a new tool. |
| **Claude Code as the agent provider** | Switch starts a Claude Code session in this `floodroom/` folder. `CLAUDE.md` is its standing instructions. |
| **Working directory = the agent's world** | The agent can only see this folder: the script, the instructions and `reports.jsonl`. |
| **Addressing with `@` and a room alias** | `!set-alias … @floodprint`. The bot stays quiet during normal chatter and acts only when called. |
| **"Who can talk to your agent: Anyone"** | Every resident in the room can report, not just the person who set it up. |
| **Photo attachments with captions** | A photo captioned `@floodprint …` reaches the agent, which can judge the depth from the picture. |
| **Shared room context** | Everyone sees every report and answer, so the chat itself becomes the public record. |

### The split between model and code

The model is good at understanding "water don reach my knee for Ada-George road" and at writing a friendly reply. It is bad at counting, timing and remembering. So:

- **Claude** (via Switch) works out what the person means, picks the command, and explains the result in one or two lines.
- **`floodroom.py`** does the exact work: matches place names ("ada-george rd" → Ada George, "mile one" → Mile 1 Diobu), stores reports, applies the 6-hour live window, marks a sighting **confirmed** once two different people report it, keeps the flood memory, and pulls the next 6 hours of rain from [Open-Meteo](https://open-meteo.com) (free, no key).

It's standard-library Python with no dependencies, and it reuses the 18 Port Harcourt neighbourhoods from FloodPrint's `config/regions/port_harcourt.json`.

## Commands the agent runs

```bash
python3 floodroom.py report "rumuola junction" "knee deep" --by Ada --note "cars turning back" --photo
python3 floodroom.py status "ada george"
python3 floodroom.py brief
python3 floodroom.py clear "Rumuola" --by Ibim
python3 floodroom.py memory "D-Line"
python3 floodroom.py places
python3 floodroom.py seed        # demo history, every record marked demo
python3 -m unittest discover -s tests
```

Depth scale: ✅ dry · 🟡 ankle (passable with care) · 🟠 knee (avoid in small cars) · 🔴 waist (do not cross) · ⛔ impassable (find another route).

## Set it up (about 10 minutes)

You need: macOS, Windows or Linux; Python 3.10+; Docker; a Claude Code account; a Telegram account with a `@username`.

1. Clone this repo. Check the script runs: `cd floodroom && python3 floodroom.py seed && python3 floodroom.py brief`.
2. Install **Switch Console** from the [Switch releases page](https://github.com/sandbox-quantum/switch/releases/latest). Open it, add a server, and choose **This computer**.
3. In Telegram, message [@BotFather](https://t.me/BotFather) and send `/newbot`. Keep the token safe.
4. Still in BotFather: `/mybots` → your bot → **Bot Settings** → **Group Privacy** → **Turn off**. Do this **before** adding the bot to a group.
5. In Switch Console: server **Home** → **Messaging apps** → **Connect** → **Telegram**. Paste the token and bot username.
6. Create a Telegram group (for example "Rumuola Estate Flood Watch"), add the bot, and send any message. The group appears under **Your Rooms**.
7. On the Telegram connection, select **Link my account…** and pick yourself.
8. Under **Your Agents**, create an agent:
   - **Agent provider:** Claude Code
   - **Working directory:** this repo's `floodroom/` folder
   - **Name:** `claude-code.floodprint.<you>`
   - **Who can talk to your agent:** **Anyone**, so every resident can report
9. In the group:
   ```text
   !invite-agent @claude-code.floodprint.<you>
   !set-alias @claude-code.floodprint.<you> @floodprint
   ```
10. Try it: `@floodprint brief`. A 👀 on your message means the agent is working.

Telegram doesn't autocomplete `@floodprint`, and it shows as plain text. That's expected; it still works.

## Demo video script (under 2 minutes)

1. **(10s) Problem.** Scroll a busy group chat: "is Rumuola flooded??" asked three times, no clear answer.
2. **(15s) Setup.** Switch Console: the agent with Claude Code in `floodroom/`, connected to the Telegram group.
3. **(60s) Live in the chat**, using two phones or two Telegram accounts:
   - `@floodprint brief`: quiet city, but "these flood often: Rumuola, D-Line".
   - Person A sends a flood photo captioned `@floodprint Rumuola junction knee deep`. It's logged as unconfirmed.
   - Person B: `@floodprint Rumuola is flooded here too`. Now it's **CONFIRMED**.
   - Person C: `@floodprint going Rumuokoro to Mile 1, which way?`. The agent checks each leg and suggests a detour.
   - `@floodprint Rumuola is dry now`. It's cleared.
4. **(15s) Why Switch.** No new app; the chat becomes the flood map; the memory grows every season.

## Limits

- The agent answers only when addressed. Switch has no scheduled messages yet, so there's no automatic morning alert.
- It runs where Switch Console runs. If the laptop sleeps, the group gets no answers. For a real estate, host Switch on a small server ([hosting docs](https://docs.switchagents.ai/switch-rooms/deploy/host-remotely)).
- Reports are community observations, not official warnings. The agent says so and points to emergency services (112).
- `reports.jsonl` is local and git-ignored, so residents' reports are never committed.

## What's next

- A linked **responders room**: confirmed waist-deep reports get passed to a second room where volunteers and the estate chairman coordinate.
- Feed confirmed reports back into the main FloodPrint app (map, hotspots, risk model).
