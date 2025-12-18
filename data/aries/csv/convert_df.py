import csv

def csv_to_tsv(csv_filename, tsv_filename):
    with open(csv_filename, 'r', newline='', encoding='utf-8') as csv_f:
        with open(tsv_filename, 'w', newline='', encoding='utf-8') as tsv_f:
            csv_reader = csv.reader(csv_f)
            tsv_writer = csv.writer(tsv_f, delimiter='\t')
            tsv_writer.writerows(csv_reader)
csv_to_tsv("aries_dev_intent.csv", "../aries_dev_intent.tsv")
csv_to_tsv("aries_train_intent.csv", "../aries_train_intent.tsv")
csv_to_tsv("aries_test_intent.csv", "../aries_test_intent.tsv")