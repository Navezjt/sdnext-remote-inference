# sdnext-remote-inference
[SD.Next](https://github.com/vladmandic/automatic) extension to send compute tasks to remote inference servers.
Aimed to be universal for all providers, feel free to request other providers.

# Providers
- SD.Next (someone else running SD.Next API)
- [Stable Horde](https://stablehorde.net/) (free, crowd computers)
- [OmniInfer](https://www.omniinfer.io/) (paid)
- other ?

# Features
|                             | SD.Next API | Stable Horde | OmniInfer |
|-----------------------------|:-----------:|:-----------:|:---------:|
| ***Model browsing***        |             |             |           |
| Checkpoints browser         |      🆗      |      ✅      |     ✅     |
| Loras browser               |      🆗      |      ⭕      |     ✅     |
| Embeddings browser          |      🆗      |      ⭕      |     🆗     |
| Lycoris browser             |      🆗      |      ❌      |     ❌     |
| Hypernet browser            |      🆗      |      ❌      |     ❌     |
| VAE selection               |      🆗      |      ❌      |     🆗     |
| ***Generation***            |             |             |           |
| From Text                   |      🆗      |      🆗      |     🆗     |
| From Image                  |      🆗      |      🆗      |     🆗     |
| Second pass (hires/refiner) |      🆗      |      🆗      |     🆗     |
| ControlNet                  |      🆗      |      🆗      |     🆗     |
| Inpainting                  |      🆗      |      🆗      |     🆗     |
| ***User***                  |             |             |           |
| Balance (credits/kudos)     |      ❌      |      🆗      |     🆗     |
| Generation cost estimation  |      ❌      |      🆗      |     🆗     |
| Image rating                |      ❌      |      🆗      |     ❌     |

✅ functional
🆗 work in progress
⭕ not needed
❌ not supported

