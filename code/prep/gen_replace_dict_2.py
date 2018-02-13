
# coding: utf-8
try:
    from ..utils.file_utils import FileSaver
    from ..utils.perf_utils import task
except SystemError as e:
    import sys
    sys.path.insert(0, '../utils/')
    from file_utils import FileSaver
    from perf_utils import task


# replace "Mr."/"Miss." with "Mr"/"Miss"

repl_sc_lst = [
    "3M®",
    "Actron®",
    "Air Lift™",
    "Armor All®",
    "Bésame Cosmetics",
    "Blitz®",
    "Blue Magic®",
    "FRAM®",  # -> "FRAM", only one labelled sample, no missing samples
    "GreatNeck®",  # -> "GreatNeck", only one labelled sample, no missing samples
    "Hypertech®",  # -> "Hypertech", only one labelled sample, no missing samples
    "Dorman®",
    "Energizer®",
    "Slime®",  # -> "Slime"
    "Streamlight®",  # -> "Streamlight"
    "Magellan®",  # -> "Magellan"
    "Mechanix Wear®",  # -> "Mechanix Wear", add brand "Mechanix and merge
    "Mercury®",  # -> "Mercury"
    "Miraclesuit®",  # -> "Miraclesuit"
    "Mobil™",  # -> "Mobil"
    "OGIO®",  # -> "OGIO"
    "Permatex®",  # -> "Permatex"
    "PlastiColor®",  # -> "PlastiColor"
    "PowerTorque®",  # add "PowerTorque" and merge
    "Preval®",  # add new brand "Preval" and merge
    "Purple Power®",  # -> "Purple Power"
    "Victorinox Swiss Army®",  # -> "Victorinox Swiss Army"
    "WD-40®",  # -> "WD-40"
]

no_repl_sc_lst = [
    "Chevron®",
    "Chroma™",
]

repl_dict = {
    "KangaRoos": "LuLaRoe",
    "Eden and Olivia": "Eden & Olivia",
    "Erica": "Erika",
    "Excell": "XCEL",
    "Feetures!": "Feetures",
    "EO Essential Oils": "EO",
    "Fiorentini + Baker": "Fiorentini",
    "Jollychic.com": "Jollychic",
    "CLEAN": "Mr Clean",
    "Mr. Clean": "Mr Clean",
    "Gerard Cos": "Gerard Cosmetics",
    "KRW": "KR3W",
    "L'eggs": "Leggs",
    "Lyons / Hit Ent.": "Lyons Hit Entertainment",
    "Lyons/Hit Ent.": "Lyons Hit Entertainment",
    "KEENS": "KEEN",
    "Per Lei": "lei",
    "Lei Lei": "lei",
    "10 Deep": "10.Deep",
    "10Deep": "10.Deep",
    "Ren & Stimpy": "Ren and Stimpy",
    "RobinRuth": "Robin Ruth",
    "Santee": "Sante",
    "Studio Mic": "Studio Microphone",
    "Mechanix": "Mechanix Wear",
    "PoGo": "PoGo! Products",
    "Poof": "Poof!",
    "Pull & Bear": "Pull&Bear",
    "Trojan Magnum": "Magnum Condom",
    "Magnum Trojan": "Magnum Condom",
    "Mattefiy": "Matteify",
    "Morgan Dollar": "Morgan Silver",
    "Silver Morgan": "Morgan Silver",
    "Oh Mamma": "Oh! Mamma",
    "Sarah Jessica Parker": "SJP by Sarah Jessica Parker",
    "PAUL and JOE": "PAUL & JOE",
    "Post it": "Post-it",
    "TV Tilt": "Tilt TV"
}

early_repl_dict = {
    "Ann": "Ann Cherry",
    "Coast": "Costa",
    "Cos": "Gerard Cos",
    "Ultra Flirt": "Flirt",
    "Lapis": "Lapis Clothing",
    "Thrive": "Thrive Lifestyle",
}

late_repl_lst = [
    "ViX",
    "Reflections",
    "Splash",
    "Split",
    "Staple",
    "me too",
    "Protege",
    "Mystic",
    "Nana",
    "No Fear",
    "Parker",
    "Presto",
    "Tano",
    "Umbra",
    "Zoom",
    "Little Mermaid",
    "PoGo",
    "Cosmetics"
]

add_lst = [
    "Eden and Olivia",
    "Eden & Olivia",
    "XCEL",
    "John Galt",
    "GK",
    "Armor All",
    "changes color",
    "Jollychic",
    "Mr Clean",
    "Gerard Cos",
    "Gerard Cosmetics",
    "Leggs",
    "Lyons/Hit Ent.",
    "KEENS",
    "Lapis Clothing",
    "Per Lei",
    "Lei Lei",
    "Kylo Ren",
    "Ren and Stimpy",
    "Ren & Stimpy",
    "Sweet Peach",
    "Thrive Lifestyle",
    "Rustic Cuff",
    "Kershaw",
    "Self Expressions",
    "Cricut Expressions",
    "Flexi Rods",
    "Flexi Lexi",
    "OSEA",
    "10 Deep",  # "10.Deep"
    "10Deep",  # "10.Deep"
    "KEENS",  # "KEEN"
    "Always kiss me",
    "Per Lei",
    "Lei Lei",
    "Barcelona Lionel",
    "Little Mermaid",
    "Winky Lux",
    "Lux De Ville",
    "Que Bella",
    "Natural Reflections",
    "Kylo Ren",
    "Ren and Stimpy",
    "Ren & Stimpy",  # "Ren and Stimpy"
    "Dark Root",
    "Robin Ruth",
    "RobinRuth",  # "Robin Ruth"
    "Twilight Saga",
    "Santee",  # "Sante"
    "Vin Scully",
    "DoTerra",
    "Nicholas Sparks",
    "Studio Makeup",
    "Studio Microphone",
    "Studio Mic",  # "Studio Microphone"
    "Sweet Peach",
    "Mechanix",  # "Mechanix Wear"
    "PoGo",  # "PoGo! Products"
    "Poof",  # "Poof!"
    "Pull & Bear",  # "Pull&Bear"
    "Cordoba",
    "Magnolia Wreath",
    "Magnolia Story",
    "Sweet Magnolia",
    "Magnum Condom",
    "Trojan Magnum",  # "Magnum Condom"
    "Magnum Trojan",  # "Magnum Condom"
    "Matteify",
    "Mattefiy"  # "Matteify"
    "Midway Arcade",
    "Miller Lite",
    "Mac Miller",
    "Moo",
    "Morgan Silver",
    "Morgan Dollar",  # "Morgan Silver"
    "Silver Morgan"  # "Morgan Silver"
    "Mystic B",
    "Daniel Tiger",
    "Troye Sivan",
    "Shakespeare",
    "Oh Mamma",  # "Oh! Mamma"
    "Sarah Jessica Parker",  # "SJP by Sarah Jessica Parker"
    "PAUL and JOE",  # "PAUL & JOE"
    "Pelican Case",
    "Complexion Perfection",
    "PING PONG",
    "Post it",  # "Post-it"
    "Cosmetics",
    "Pura Vida",
    "Tilt TV",
    "TV Tilt"  # "Tilt TV"
    "Attack on Titan",
    "Toostie Roll",
    "Walter White",
    "Most Wanted",
    "Avo Wanted",
    "Azzaro Wanted",
    "Fiesta Ware",
    "Russell Westbrook",
    "Bring me the Horizon",
    "Forza Horizon"
]


@task("save replace list/dict for preprocessing")
def main():
    fs = FileSaver('./final/')
    fs.save_list(repl_sc_lst, "brand.repl_spec_char")
    fs.save_list(no_repl_sc_lst, "brand.no_repl_spec_char")
    fs.save_list(add_lst, "brand.add")
    fs.save_list(late_repl_lst, "brand.late_repl")
    fs.save_dict(repl_dict, "brand.repl")
    fs.save_dict(early_repl_dict, "brand.early_repl")


if __name__ == "__main__":
    main()
