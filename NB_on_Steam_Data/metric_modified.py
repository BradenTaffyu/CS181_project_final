"""
F-Score metrics for testing classifier, also includes functions for data extraction.
Author: Vivek Narayanan
"""
import os
from info import MyDict
import info, pickle


def get_paths(game_name):
    """
    Returns supervised paths annotated with their actual labels.
    """
    files = "Data/" + game_name + ".csv" 
    
    return files 

# Start of Selection
def fscore(classifier, csv_path):
    # Start Generation Here
    import csv
    # Read existing rows and compute predictions
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        # Add 'predicted' to the list of fieldnames
        fieldnames = reader.fieldnames + ['predicted']
        for row in reader:
            review_text = row.get('review', '')
            # Compute and store prediction
            row['predicted'] = classifier(review_text)
            rows.append(row)
    # Write back with the new 'predicted' column
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    # End Generation Her
    # import csv
    # with open(csv_path, 'r', encoding='utf-8') as csvfile:
    #     reader = csv.DictReader(csvfile)
    #     for row in reader:
    #         review_text = row.get('review', '')
    #         result = classifier(review_text)
    #         print(result)
    #             # End of Selectio
        

    # prec = 1.0 * tpos / (tpos + fpos)
    # recall = 1.0 * tpos / (tpos + fneg)
    # f1 = 2 * prec * recall / (prec + recall)
    # accu = 100.0 * (tpos + tneg) / (tpos+tneg+fpos+fneg)
    # print ("True Positives: %d\nFalse Positives: %d\nFalse Negatives: %d\n" % (tpos, fpos, fneg))
    # print ("Precision: %lf\nRecall: %lf\nAccuracy: %lf" % (prec, recall, accu))
    # print("tpos,tneg,fpos, fneg", tpos, tneg, fpos, fneg)

def main():
    from info import classify, train 
    with open("reduceddata.pickle", "rb") as f:
        info.pos, info.neg, info.totals = pickle.load(f)
    # 还要把 features 从 info.feature_selection_trials 里拿出或者重跑一次
    info.features = set(info.pos.keys())  

    game_names = [
        "Grand Theft Auto V",
        "Tom Clancy's Rainbow Six® Siege",
        "Counter-Strike 2",
        "Dead by Daylight",
        "Call of Duty®: Black Ops III",
        "Sea of Thieves: 2024 Edition",
        "ELDEN RING",
        "Total War: WARHAMMER III",
        "Warframe",
        "Call of Duty®",
        "Apex Legends™",
        "Noita",
        "Wallpaper Engine",
        "Dragon's Dogma 2",
        "NARAKA: BLADEPOINT"
    ]
    for game_name in game_names:
        paths = get_paths(game_name.replace(' ', '_').replace('®', '').replace(':', '').replace('™', ''))
        fscore(info.classify, paths)

if __name__ == '__main__':
    main()
