# sdnext-remote-inference
SD.Next extension to send compute tasks to remote inference servers.
Aimed to be universal for all providers, feel free to request other providers.

# Providers
- [SD.Next](https://github.com/vladmandic/automatic) (someone else running SD.Next API)
- [Stable Horde](https://stablehorde.net/) (free, crowd computers)
- [OmniInfer](https://www.omniinfer.io/) (paid)
- other ?

# Features
|                             | SD.Next API | Stable Horde | OmniInfer  |
|-----------------------------|:-----------:|:------------:|:----------:|
| ***Model browsing***        |             |              |            |
| Checkpoints browser         |      ✅     |      ✅      |     ✅     |
| Loras browser               |      ✅     |      ⭕      |     ✅     |
| Embeddings browser          |      ✅     |      ⭕      |     ✅     |
| Hypernet browser            |      🆗     |      ❌      |     ❌     |
| VAE browser                 |      🆗     |      ❌      |     🆗     |
| ***Generation***            |             |              |            |
| From Text                   |      🆗     |      ✅      |     🆗+    |
| From Image                  |      🆗     |      ✅      |     🆗     |
| Inpainting                  |      🆗     |      ✅      |     🆗     |
| Second pass (hires/refiner) |      🆗     |      ✅      |     🆗     |
| Loras and TIs               |      🆗     |      🆗      |     🆗     |
| ControlNet                  |      🆗     |      🆗      |     🆗     |
| Upscale                     |      🆗     |      🆗      |     🆗     |
| ***User***                  |             |              |            |
| Balance (credits/kudos)     |      ❌     |      ✅      |     ✅     |
| Generation cost estimation  |      ❌     |      🆗      |     🆗     |
| Image rating                |      ❌     |      🆗      |     ❌     |

- ✅ functional
- 🆗+ work in progress
- 🆗 planned
- ⭕ not needed
- ❌ not supported

## Additional features
- Stable Horde worker settings
- API calls caching
- Hide NSFW networks option (Stable Horde / OmniInfer)

# Installation & usage
1. Launch SD.Next with `--no-download` option
2. Installation
    1. Go to extensions > manual install > paste `https://github.com/BinaryQuantumSoul/sdnext-remote-inference` > install
    2. Go to extensions > manage extensions > apply changes & restart server
    3. Go to system > settings > remote inference > set right api endpoints & keys
3. Usage
    1. Select desired remote inference service in dropdown, refresh model list and select model
    2. Set generations parameters as usual and click generate

# Credits
Inspired by:
- [natanjunges/stable-diffusion-webui-stable-horde](https://github.com/natanjunges/stable-diffusion-webui-stable-horde)
- [omniinfer/sd-webui-cloud-inference](https://github.com/omniinfer/sd-webui-cloud-inference)
