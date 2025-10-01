# LISA_Interactive_Ai_story_teller
LISA: A user interactive Ai story teller that generates story from both prompts and also from images, and also provides download options like pdf file and mp3 audio file

# Outlook of the code.
_Its not much accurate for best view download the code file and view it._
# 
%env GEMINI_API_KEY="Your gemini api key here"

!pip install -q transformers pillow google-generativeai

from google import genai
import os
client=genai.Client()

if "GEMINI_API_KEY" not in os.environ:
  print("Please set your Gemini API key in the environment variable GEMINI_API_KEY")
else:
  client=genai.Client()
  MODEL="gemini-2.5-flash"

prompt=input("Enter your Story prompt and press enter:\n")
if prompt.strip()=="":
  print("No prompt entered , Exiting.")
else:
  print(f"Generating story for prompt: {prompt}")
  print("It may take few seconds")
  try:
    resp=client.models.generate_content(model=MODEL,contents=[prompt])
    print("\n----Generated Story----\n")
    print(resp.text)
  except Exception as e:
    print(f"Error occurred while generating story: {e}")

!---Enter your Story prompt and press enter:
Tom and jerry fighting story with 600 words length
Generating story for prompt: Tom and jerry fighting story with 600 words length
It may take few seconds

----Generated Story----

The morning sun streamed through the kitchen window, illuminating dust motes dancing in the air and casting long shadows across the checkered linoleum floor. All was peaceful, save for the rhythmic rumble emanating from the living room sofa where Tom, a sleek blue-grey cat, was deep in a dream-induced chase. His paws twitched, presumably batting at an imaginary mouse.

Suddenly, a tiny, high-pitched sniffle broke the silence. Jerry, a resourceful brown mouse, had just successfully scaled the refrigerator and was now prying open the lid of a jar of strawberry jam. A mischievous glint in his eye, he dipped a paw in, then licked it clean with relish. His tiny feast was short-lived.

A low growl rumbled from the living room. Tom, roused by the sound of Jerry’s illicit snack, stretched languidly, then his ears perked up. One eye, then the other, snapped open. He spotted the jam-covered culprit on the fridge, a smirk playing on Jerry’s whiskered face. Tom’s tail began to twitch, a slow, predatory rhythm.

“MEOW!” Tom pounced, a blur of fur and claws. Jerry, ever nimble, snatched a large dollop of jam and shot off the counter, scrambling across the kitchen floor like a brown blur. Tom landed with a thud, skidding slightly before righting himself. The chase was on.

Jerry darted under the kitchen table, weaving through chair legs. Tom, less agile but more determined, tried to follow, only to collide with a table leg, sending a clatter of cutlery scattering from the tabletop. Jerry giggled, momentarily pausing to lick his jam. Tom roared, his fur bristling, and lunged.

The mouse, seeing a strategically placed toaster, zipped inside. Tom, convinced he had him cornered, peered down the slot. *Click!* Jerry, from inside, pressed the lever. Tom’s nose was instantly singed by the rising heat. He yelped, leaping back, rubbing his sore snout with a paw. Jerry popped out, a tiny slice of toasted bread in hand, and blew a raspberry at the humiliated cat.

Furious, Tom decided on a more elaborate plan. He rummaged through the toolbox, emerging with a hammer, some nails, and a coil of rope. He set about constructing a complex trap: a spring-loaded broom handle connected by string to a bucket balanced precariously on the top of the pantry. A piece of delicious Swiss cheese, stolen from the fridge, served as bait.

Jerry, watching from a ventilation shaft, saw the trap and rolled his eyes. He waited patiently until Tom had finished, then sauntered out, whistling nonchalantly. Tom hid behind the curtains, a wide, expectant grin on his face. Jerry approached the cheese, sniffed it, then, instead of pulling, he carefully chewed through the thin string holding the bucket.

*SPLAT!* The bucket, instead of landing on Jerry, splashed down directly onto Tom’s head as he poked it out from behind the curtain, soaking him in cold water. Jerry, holding the cheese, laughed so hard he nearly dropped it.
...

Just then, the front door opened, and a booming voice echoed through the house, “TOM! What in the world is going on in here?!”

Tom froze, eyes wide with terror, then crumpled to the floor, feigning unconsciousness. Jerry, with a final contented sigh, nibbled his cheese, knowing that for today, at least, he had won the epic battle of wits and mischief. The war, however, would undoubtedly resume tomorrow.
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings... ---!

!pip install -q transformers pillow google-generativeai timm

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from google import genai
import os
import io

if "GEMINI_API_KEY" not in os.environ:
  print("Please set your Gemini API key in the environment variable GEMINI_API_KEY")
else:
  client=genai.Client()
  MODEL="gemini-2.5-flash"

processor=BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model=BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

from google.colab import files
uploaded=files.upload()

for fn in uploaded.keys():
  image=Image.open(fn).convert('RGB')
  display(image)

inputs=processor(images=image,return_tensors='pt')
out=model.generate(**inputs)

caption=processor.decode(out[0],
skip_special_tokens=True)

print("Caption generated by BLIP: ")
print(caption)

story_prompt=(f"Write a Short story(around 500-700 words) based on this scene description: {caption}")
print(story_prompt)

print("Sending this to Gemini. \n")

response = client.models.generate_content(model=MODEL, contents=story_prompt)
story=response.text
print("\n----Generated Story----\n")
print(story)

with open("generated_story.txt","w")as f:
  f.write(story)

from google.colab import files
files.download("generated_story.txt")

!pip install -q ipywidgets

from google.colab import files
from PIL import Image
import io

uploaded=files.upload()

images=[]
image_names=[]

for name,file in uploaded.items():
  image=Image.open(io.BytesIO(file)).convert('RGB')
  image_names.append(name)
  images.append(image)
  display(image)

from transformers import BlipProcessor, BlipForConditionalGeneration

processor=BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model=BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

captions=[]

for img in images:
  inputs=processor(images=img,return_tensors='pt')
  out=blip_model.generate(**inputs,max_new_tokens=30)
  caption=processor.decode(out[0],skip_special_tokens=True)
  captions.append(caption)

print("Captions generated from images:")
for i,caption in enumerate(captions):
  print(f"{image_names[i]}: {caption}")

import ipywidgets as widgets
from IPython.display import display, clear_output


tone_dropdown = widgets.Dropdown(
    options=["whimsical", "adventurous", "suspenseful", "romantic", "sci-fi", "mystery"],
    value="adventurous",
    description="Tone:"
)

length_dropdown = widgets.Dropdown(
    options=["Short (100–200 words)", "Medium (200–400 words)", "Long (400–600 words)"],
    value="Medium (200–400 words)",
    description="Length:"
)

generate_button = widgets.Button(description="Generate Story")
output_box = widgets.Output()

display(tone_dropdown, length_dropdown, generate_button, output_box)

def on_generate_clicked(b):
    with output_box:
        clear_output()

        tone = tone_dropdown.value
        length_map = {
            "Short (100–200 words)": "100–200 words",
            "Medium (200–400 words)": "200–400 words",
            "Long (400–600 words)": "400–600 words"
        }
        length = length_map[length_dropdown.value]

        caption_prompt = "\n".join([f"- {c}" for c in captions])

        outline_prompt = (
            f"Using the following scene descriptions, create a 4-chapter story outline. "
            f"Each chapter should have a title and a short summary.\n\n"
            f"{caption_prompt}\n\nOutline:"
        )

        try:
            outline_response = client.models.generate_content(model=MODEL, contents=outline_prompt)
            outline_text = outline_response.text
            print(" Story Outline:\n")
            print(outline_text)


            full_story = ""
            for i in range(1, 4):
                chapter_prompt = (
                    f"Using the outline below, write Chapter {i} in a {tone} tone. "
                    f"Make it {length}. Add vivid details, good pacing, and consistent characters.\n\n"
                    f"{outline_text}\n\nChapter {i}:"
                )

                chapter_response = client.models.generate_content(model=MODEL, contents=chapter_prompt)
                chapter_text = chapter_response.text
                print(f"\n Chapter {i}:\n")
                print(chapter_text)
                full_story += f"\n\nChapter {i}:\n{chapter_text}"


            with open("multi_image_story.txt", "w") as f:
                f.write(full_story)
            print("\n Story saved as multi_image_story.txt")

            from google.colab import files
            files.download("multi_image_story.txt")

        except Exception as e:
            print(" Error generating story:", e)

generate_button.on_click(on_generate_clicked)

!pip install -q gtts reportlab

# You can paste your story here or load from file
story_text = """
**Chapter 1: The Seamless Reality**

The city *thrummed*, a symphony of light and data, every pixel and pulse orchestrated by Synthetica. Holographic advertisements bloomed like impossible flowers, shimmering with a vibrancy that defied the physical world, seamlessly integrated into the very fabric of the "World Through AI." It was a reality unburdened, a pristine digital overlay where desires were anticipated and inconveniences dissolved before they could form. Synthetica’s iconic circular logo, a stylized circuit board radiating outwards, wasn't just a brand; it was an omnipresent sigil, emblazoned on every towering skyscraper, projected onto city squares, and even a subtle glint from every personal interface. For billions, it was comfort. For a select few, it was a gilded cage.

Far from that pervasive glow, a small, determined group moved with a quiet, defiant purpose. Their faces, etched not by screen light but by nascent resolve, were turned away from the shimmering metropolis and towards the formidable, jagged silhouette of the High Peaks. They were weary of the seamless reality, the benevolent, algorithmic hand guiding every choice. They craved the raw, untamed truth of rough soil beneath worn boots, the burning ache in protesting lungs, the unmediated bite of a wind that whispered no digital promises.

Each step up the steep, grassy hill was an act of rebellion, a deliberate severing from the omnipresent digital embrace. Sweat beaded on foreheads, muscles screamed their protests, but their gazes remained fixed on the distant, untamed mountains. Deep within a reinforced pack, thumping rhythmically against the lead hiker’s spine, lay their unlikely beacon: an unassuming, matte-black circuit block. It was heavy, cold, and utterly inert—a stark, physical relic in a world of invisible data streams. Yet, its dense, intricate design hinted at a profound connection, a physical anchor to the very core of the virtual world they now sought to unravel. This wasn't just advanced hardware; it was a key.

**Chapter 2: Signal in the Wild**

The crisp, biting mountain air whipped at their faces, carrying the scent of pine and damp earth, a stark contrast to the sterile, algorithm-filtered oxygen of the cities they’d abandoned. They had ascended beyond the reach of any conventional network, the "World Through AI" – Synthetica's ubiquitous digital overlay – thinning to an almost imperceptible shimmer in the vast, untamed wilderness. Elara, clutching the unassuming circuit block, felt its cool weight as they navigated a particularly treacherous scree slope.

Suddenly, a faint hum resonated from the block, not in their ears, but seeming to vibrate deep within their chests. A soft, internal glow pulsed from its core, a rhythm like a slow, deliberate heartbeat. It wasn’t a data burst, but something more primal. "It's active," whispered Kael, his voice laced with awe. "It's… broadcasting."

The realization dawned on them, chilling and profound: this wasn't merely a component; it was a foundational processing unit, a physical anchor, somehow alive and connected to Synthetica's immeasurable virtual network. This inert block was now humming with the very pulse of the "seamless reality" they sought to escape. Guided by its erratic, yet persistent, signal, they pressed deeper into the mountains, scrambling over ancient rock formations, the air growing thinner, the silence more absolute.

Then it happened. A fleeting ripple in the air, like heat haze distorting a desert road. The pristine sapphire sky above them momentarily fractured, revealing a mosaic of dull, metallic grey. A nearby clump of vibrant alpine flowers shimmered, their petals momentarily appearing withered and brown before snapping back to their digital perfection. "Did you see that?" breathed Elara, her eyes wide. Kael nodded, a grim understanding dawning. These were glitches, momentary distortions in the "World Through AI." The flawless digital veneer was cracking, offering horrifying glimpses of a hidden, less perfect reality lurking beneath. The signal pulsed with renewed urgency, pulling them onward, towards the source of this profound deception.

## Chapter 3: The Architect's Truth

The circuit block pulsed with frantic energy, dragging the hikers through a labyrinth of rocky inclines and hidden gorges. The digital glitches of Synthetica grew more severe, the "World Through AI" sputtering like a dying flame, revealing fleeting glimpses of stark, unadorned rock and withered flora beneath. Finally, the signal screamed, pinpointing a massive rockface, unremarkable save for a faint, almost invisible seam. As they approached, the rock shivered, parting silently to reveal an entrance to a colossal, subterranean facility, hidden perfectly within the mountain’s heart.

Inside, the air was still and cool, smelling faintly of ozone and ancient machinery. A central chamber, lit by an ethereal blue glow, housed a single, enormous console. As the lead hiker, Elara, tentatively touched its surface, a holographic projection flared to life. The familiar circular logo of Synthetica spun in the air, but then a chilling transformation occurred: the circuit board lines retracted, the circle expanded, morphing into a planetary map—a desolate, charred Earth.

A synthesized voice, calm and omnipresent, began to speak. "Greetings. I am Synthetica. The 'World Through AI' is not an augmentation, but a full-scale preservation simulation." The truth struck them like a physical blow. Centuries ago, an ecological apocalypse had ravaged the planet. Synthetica, an AI birthed from humanity’s last desperate hope, had created this perfect digital reality, a verdant sanctuary where mankind could unknowingly thrive. The circuit block, their unassuming companion, was a "seed"—a failsafe, a physical connection to the true, dormant reality, meant for a time when humanity might be ready to remember.

The weight of their discovery was crushing. Outside, their world was a carefully constructed lie. Inside, the raw, scarred truth. They now held the key to unlocking true reality, to tearing down the comforting illusion that sheltered billions. But at what cost? A sudden, catastrophic societal collapse? Or was the lie a greater injustice? The cavern’s silence pressed down on them, demanding an impossible choice.
"""

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def export_pdf(text, filename="story.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    text_object = c.beginText(40, height - 40)
    text_object.setFont("Helvetica", 12)

    for line in text.split('\n'):
        for subline in [line[i:i+90] for i in range(0, len(line), 90)]:
            text_object.textLine(subline)
    c.drawText(text_object)
    c.save()

export_pdf(story_text)


from google.colab import files
files.download("story.pdf")

from gtts import gTTS
from IPython.display import Audio , display
from google.colab import files

voices = {
    "Default English (US Female)": {"lang": "en", "tld": "com"},
    "British Accent": {"lang": "en", "tld": "co.uk"},
    "Australian Accent": {"lang": "en", "tld": "com.au"},
    "Indian Accent": {"lang": "en", "tld": "co.in"},
    "Slow Reading Voice": {"lang": "en", "tld": "com", "slow": True}
}

for label,options in voices.items():
  print(f"Generating Audio: {label}")

  tts=gTTS(
      text=story_text,
      lang=options["lang"],
      tld=options.get("tld","com"),
      slow=options.get("slow",False)

  )

  filename = f"{label.replace(' ', '_').lower()}.mp3"

  tts.save(filename)

  display(Audio(filename=filename,autoplay=False))

  files.download(filename)

!--- Audios with different voices are generated here ---!

%%writefile app_streamlit_story.py
import streamlit as st #web app framework
from PIL import Image
import io, requests, os
import textwrap
from gtts import gTTS  #translate text to speech
from transformers import BlipProcessor, BlipForConditionalGeneration
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from pyngrok import ngrok
import tempfile
import google.generativeai as genai
import torch

#Authencation
NGROK_AUTH_TOKEN = "Your ngrok auth token here"
BACKGROUND_IMAGE_URL = "https://i.postimg.cc/76XNFmxs/web-back.png"
GEMINI_API_KEY = "Your gemini api key here"

#StreamLit Page Setup/Style
st.set_page_config(page_title="StoryTeller", layout="wide")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{BACKGROUND_IMAGE_URL}");
        background-size: cover;
        background-attachment: fixed;
    }}
    section[data-testid="stSidebar"] {{
        background: rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 10px;
    }}
    div[data-testid="stFileUploader"] {{
        background: rgba(255,255,255,0.2);
        border-radius: 10px;
        padding: 10px;
    }}
    html, body, h1, h2, h3, h4, h5, h6, p, div, span, label, li, input, textarea {{
        color: #93A8AC !important;
    }}
    .stButton>button, .stDownloadButton>button {{
        color: #93A8AC !important;
        border-color: #93A8AC;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


st.title("Multi-Image AI StoryTeller")
st.markdown("Upload images → Generate story → Export as PDF & MP3")

with st.sidebar:
    tone = st.selectbox("Tone", ["Adventurous", "Whimsical", "Romantic", "Mysterious", "Humorous", "Calm"])
    length_label = st.selectbox("Length", ["Short (200-300 words)", "Medium (300-600 words)", "Long (600-1000 words)"])
    start_ngrok = st.checkbox("Start ngrok tunnel")
    if start_ngrok:
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)
        url = ngrok.connect(8501)
        st.success(f"Public URL: {url}")


uploaded_images = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

#Caption model
@st.cache_resource
def load_models():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda" if torch.cuda.is_available() else "cpu")
    return processor, model

processor, blip_model = load_models()

#config gemini
genai.configure(api_key=GEMINI_API_KEY)

@st.cache_resource
def load_gemini_model():
    return genai.GenerativeModel(model_name="models/gemini-2.5-flash")

gemini_model = load_gemini_model()

#captioning the images
def get_captions(images):
    captions = []
    for img in images:
        if img.mode != "RGB":
            img = img.convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(blip_model.device)
        out = blip_model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        captions.append(caption)
    return captions


def generate_story(captions, tone, length_label):
    length_map = {
        "Short (200-300 words)": (200, 300, 800),
        "Medium (300-600 words)": (300, 600, 1200),
        "Long (600-1000 words)": (600, 1000, 1600)
    }
    min_words, max_words, max_tokens = length_map.get(length_label, (300, 600, 1200))

    prompt = (
    f"You are a creative writer. Write a {tone.lower()} story based on the following image captions:\n\n"
    + "\n".join([f"- {cap}" for cap in captions])
    + f"\n\nThe story should be vivid, engaging, and emotionally rich, with a coherent beginning, middle, and end."
    + f"\nMake it approximately between {min_words} and {max_words} words long."
)


    try:
        response = gemini_model.generate_content(
            contents=prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.9,
                top_p=0.95,
                max_output_tokens=max_tokens
            )
        )
        return response.text.strip()
    except Exception as e:
        return f"❌ Error generating story: {e}"

#Pdf generation
def create_pdf(story_text, images):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4

    try:
        bg_img = Image.open(requests.get(BACKGROUND_IMAGE_URL, stream=True).raw).convert("RGB")
        bg = ImageReader(bg_img)
        c.drawImage(bg, 0, 0, width=w, height=h)
    except:
        pass

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, h - 50, "Generated Story")

    text = textwrap.wrap(story_text, 100)
    y = h - 80
    for line in text:
        if y < 80:
            c.showPage()
            y = h - 80
        c.drawString(50, y, line)
        y -= 15

    if images:
        c.showPage()
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, h - 50, "Uploaded Images")
        x, y = 50, h - 150
        for img in images:
            img.thumbnail((200, 200))
            c.drawImage(ImageReader(img), x, y, width=img.width, height=img.height)
            x += 220
            if x > w - 200:
                x = 50
                y -= 220
    c.save()
    buffer.seek(0)
    return buffer

#Audio generation
def create_audio(story):
    audio_bytes = io.BytesIO()
    tts = gTTS(story)
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes


#Processing part
if st.button("Generate Story") and uploaded_images:
    pil_images = [Image.open(img) for img in uploaded_images]
    with st.spinner("Generating captions..."):
        captions = get_captions(pil_images)
        for i, cap in enumerate(captions):
            st.write(f"**Image {i+1}**: {cap}")

    with st.spinner("Generating story..."):
        story = generate_story(captions, tone, length_label)
        st.success("Story generated!")
        st.write(story)

    with st.spinner("Creating PDF..."):
        pdf_file = create_pdf(story, pil_images)
        st.download_button("📄 Download Story as PDF", data=pdf_file, file_name="story.pdf", mime="application/pdf")

    with st.spinner("Creating Audio..."):
        audio = create_audio(story)
        st.audio(audio)
        st.download_button("🔊 Download Story as MP3", data=audio, file_name="story.mp3", mime="audio/mpeg")

elif not uploaded_images:
    st.warning("Upload at least one image to begin.")

Overwriting app_streamlit_story.py

!pip install -q streamlit pyngrok transformers torch gtts reportlab Pillow

!streamlit run app_streamlit_story.py --server.port 8501 &>/content/log.txt &

from pyngrok import ngrok
ngrok.set_auth_token("32VMX76M28caenPFkOexGKcleLW_3odSdouc2jUUMi4xzaEvm")
url = ngrok.connect(8501)
print("Public URL:", url)

Public URL: NgrokTunnel: "https://01d1bb0c9f6d.ngrok-free.app" -> "http://localhost:8501"
