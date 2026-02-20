Transformer Model za bilingvalni prevod teksta

Ovaj projekat implementira model zasnovan na Transformer arhitekturi za bilingvalno prevođenje, dizajniran za prevođenje teksta sa 
jednog jezika na drugi koristeći sekvencijalnu arhitekturu. Model je napravljen koristeći PyTorch i koristi tehnike tokenizacije za 
efikasno upravljanje ulaznim tekstom. Uključuje funkcionalnosti za učitavanje podataka, prethodnu obradu, obuku modela i evaluaciju.

Karakteristike:
Projekat koristi 'opus_books' set podataka za obuku i validaciju modela za prevođenje, automatski preuzet 
prilikom pokretanja sa sledećeg sajta: https://huggingface.co/datasets/Helsinki-NLP/opus_books.
Implementira Transformer model, uključujući komponente enkodera i dekodera.
Koristi tokenizator na osnovu reči za konverziju teksta u tokene, omogućavajući modelu da razume i obradi različite jezike.
Omogućava vizualno praćenje obuke metrika, kao što je gubitak tokom procesa obuke.
Nakon svake epohe, performanse modela se ocenjuju korišćenjem metrika kao što su stopa grešaka karaktera (CER) i stopa grešaka reči (WER).

Da bi započeli obuku Transformer modela, potrebno je run-ovati train.py fajl, ili sledeću komandu u terminalu:
  python train.py


    
