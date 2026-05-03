# Blurrinator - plan implementacji

## Cel produktu

Blurrinator jest prostą web appką do redakcji prywatności w filmach YouTube, uruchamianą na dedykowanej stacji roboczej i dostępną z zewnątrz przez przekierowanie portów. Operator wrzuca film, system automatycznie wykrywa twarze i osoby, zostawia niezasłonięte tylko osoby z allowlisty, nakłada estetyczny mocny blur na pozostałe twarze lub przewidywane obszary głowy, oznacza niepewne momenty do review i eksportuje finalny plik MP4.

System działa na jednej stacji roboczej z GPU Nvidia Blackwell 96 GB VRAM. Wolumen zakładany dla pierwszej wersji to około jeden film tygodniowo, więc priorytetem jest pewność, kontrola operatora i jakość eksportu, a nie obsługa wielu równoległych klientów.

## Założenia decyzyjne

- Aplikacja działa na dedykowanej stacji roboczej i jest dostępna z internetu przez przekierowanie portu na routerze.
- Oryginalny materiał wideo nie opuszcza maszyny.
- Domyślna polityka prywatności jest konserwatywna: każda osoba lub twarz bez pewnego dopasowania do allowlisty jest zasłaniana.
- Dla osób bez wykrytej twarzy system blurrowuje przewidywany obszar głowy i oznacza fragment jako wymagający review.
- Przypadkowe zablurowanie youtubera jest akceptowalne tylko jako błąd do poprawienia w review, nigdy jako powód do agresywnego odblurrowania niepewnej osoby.
- Dzieci nie wymagają osobnego klasyfikatora wieku w pierwszej wersji, ponieważ wszystkie osoby spoza allowlisty są zasłaniane.
- Audio zostaje przygotowane architektonicznie pod przyszłą redakcję głosu, ale pierwsza wersja nie modyfikuje ścieżki audio.
- Pierwszy efekt redakcji to estetyczny mocny blur; architektura efektów ma pozwalać później na nakładki sezonowe, takie jak twarz Mikołaja lub zajączek.

## Licencje i wybór modeli

W projekcie stosujemy tylko komponenty z licencjami open source lub modelami opublikowanymi z jawną licencją pozwalającą na lokalne użycie produkcyjne. Każdy model trafia do rejestru modeli z nazwą, wersją, źródłem, licencją, hashami plików i datą pobrania.

Kod aplikacji publikujemy publicznie na AGPL-3.0. To pozwala użyć Ultralytics YOLO bez licencji Enterprise, pod warunkiem że całe źródło aplikacji, modyfikacje, skrypty uruchomieniowe, konfiguracja buildów oraz kod integrujący modele zostaną opublikowane na licencji zgodnej z AGPL-3.0. Materiały wideo, enrollment osób, embeddingi, prywatne sekrety, lokalne konfiguracje operatora i artefakty analizy pozostają danymi prywatnymi, a nie częścią kodu źródłowego.

Rekomendowany zestaw startowy:

- OpenCV 4.5+ jako bazowa biblioteka CV, ponieważ OpenCV publikuje wersje 4.5.0 i nowsze na Apache 2.0: https://opencv.org/license/
- OpenCV Zoo YuNet do detekcji twarzy, ponieważ katalog modelu jest opisany jako MIT: https://huggingface.co/opencv/face_detection_yunet
- OpenCV Zoo SFace do embeddingów twarzy, ponieważ model jest częścią OpenCV Zoo z osobnym plikiem licencji w katalogu modelu: https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface
- Ultralytics YOLO11 detect do wykrywania osób: https://www.ultralytics.com/license
- Ultralytics YOLO11-seg do segmentacji sylwetek, gdy potrzebujemy maski osoby zamiast samego boxa.
- Ultralytics YOLO11-pose do keypointów i przewidywania obszaru głowy przy niewidocznej twarzy.
- Ultralytics tracking do łączenia detekcji osób między klatkami; tracker zapisujemy w model registry razem z konfiguracją.
- FFmpeg do dekodowania i renderu wideo, z weryfikacją licencji builda używanego w dystrybucji lokalnej.

Warunki użycia Ultralytics:

- Utrzymujemy publiczne repozytorium z kompletnym kodem odpowiadającym uruchamianej wersji.
- Ultralytics deklaruje, że ich kod, modele, pipeline treningowy oraz modele trenowane/fine-tunowane w ich ekosystemie wymagają AGPL-3.0 albo licencji Enterprise: https://www.ultralytics.com/license
- Jeśli użyjemy Ultralytics, model registry musi zapisać nie tylko hash wag, ale też informację, czy są to wagi pretrained, fine-tuned czy trenowane lokalnie. Fine-tuned weights traktujemy jako artefakt objęty obowiązkami AGPL zgodnie z deklaracją Ultralytics.
- Wariant YOLO upraszcza wybór modelu osoby, bo Ultralytics ma spójny ekosystem detekcji, segmentacji, pose i trackingu. Nadal wymaga testów jakościowych, bo licencja nie rozwiązuje ryzyka pominiętej twarzy lub błędnego trackingu.

Alternatywa awaryjna dla rozpoznawania twarzy:

- dlib + `dlib_face_recognition_resnet_model_v1.dat`, jeśli SFace będzie za słaby na materiałach testowych. Modele dlib są opublikowane jako CC0/public domain w repozytorium `davisking/dlib-models`: https://github.com/davisking/dlib-models

Komponenty odrzucone jako domyślne:

- InsightFace, ponieważ jego publiczne modele mają ograniczenia licencyjne dla użycia komercyjnego według README projektu.
- Klasyfikatory wieku, płci lub innych cech wrażliwych, ponieważ nie są potrzebne do celu produktu i zwiększają ryzyko prawne oraz produktowe.

## Architektura systemu

System składa się z pięciu warstw:

- Aplikacja Django: widoki HTML dla operatora (upload, allowlista, projekt, review, eksport), autoryzacja lokalna, konfiguracja projektu, stan jobów, zapis decyzji operatora.
- Worker wideo: osobny proces tej samej aplikacji Django, dekodowanie, inferencja modeli, tracking, generowanie masek, render finalny.
- Storage lokalny: oryginały, artefakty analizy, decyzje review, eksporty, audyt - katalog poza repo, ścieżka w `settings.py`.
- Model registry: jawna ewidencja modeli, licencji, wersji i hashy w bazie Django.

Rekomendowany stack:

- Backend: Python 3.12, Django 5.x, Django ORM, Django forms i widoki klasowe. Bez podziału na osobne API i frontend - widoki Django renderują strony, a interaktywność po stronie przeglądarki dorzucamy minimalnym JS (HTMX lub Alpine, gdy faktycznie potrzebne, np. w odtwarzaczu review).
- Worker: ten sam projekt Django uruchamiany jako osobny proces przez management command (`python manage.py run_worker`). Kolejka jobów w bazie przez Django-Q2 albo Huey w trybie SQLite/Postgres - bez Redisa. Worker korzysta z Ultralytics YOLO na PyTorch dla osób, segmentacji, pose i trackingu, ONNX Runtime GPU lub OpenCV DNN dla modeli twarzy, FFmpeg przez kontrolowany proces systemowy.
- Baza: SQLite na start, bo to lokalny serwer dla jednego operatora. Migracja na lokalnego PostgreSQL dopiero gdy SQLite zacznie blokować równoległe zapisy z workera (Django ułatwia tę zmianę przez ustawienie `DATABASES`).
- Uruchomienie: lokalna instalacja Pythona, sterowniki NVIDIA i CUDA na hoście, `pip install -r requirements.txt` w venv, `python manage.py migrate`, `python manage.py runserver` plus drugi terminal z `python manage.py run_worker`. Bez Dockera, bez Compose, bez NVIDIA Container Toolkit - GPU jest używane bezpośrednio przez PyTorch.
- Ekspozycja zewnętrzna: bezpośrednie wystawienie portu Django (`runserver` w trybie roboczym, Gunicorn dopiero gdy zacznie być potrzebny) przez port forwarding na routerze.

## Model danych

Encje domenowe:

- `Project`: pojedynczy materiał wideo i jego konfiguracja.
- `SourceVideo`: metadane oryginału, ścieżka lokalna, checksum, czas trwania, FPS, rozdzielczość, codec.
- `AllowedPerson`: osoba z allowlisty, zgody, nazwa ekranowa, status aktywny.
- `EnrollmentImage`: zdjęcie lub crop twarzy użyty do enrollmentu.
- `FaceEmbedding`: embedding twarzy przypisany do osoby z allowlisty, wersja modelu, jakość próbki.
- `AnalysisJob`: status analizy, postęp, błędy, użyte modele.
- `Detection`: surowa detekcja twarzy, osoby, pozy lub maski w konkretnej klatce.
- `Track`: ciąg detekcji tej samej twarzy lub osoby między klatkami.
- `IdentityDecision`: automatyczna albo ręczna decyzja, czy track jest allowlisted, blurred, uncertain.
- `RedactionRegion`: finalny region do zasłonięcia w czasie, z geometrią, efektem i powodem.
- `ReviewFlag`: moment wymagający uwagi operatora.
- `ExportJob`: render finalnego pliku, parametry kodeka, ścieżka eksportu.
- `AuditEvent`: kto i kiedy podjął decyzję, zmienił allowlistę, wyeksportował film albo usunął dane.

Geometria regionów jest zapisywana w koordynatach obrazu źródłowego. Dla twarzy używamy elipsy lub zaokrąglonego prostokąta rozszerzonego o margines. Dla fallbacku głowy używamy prostokąta lub elipsy wyliczonej z pozy, proporcji sylwetki i historii tracka.

## Pipeline analizy wideo

1. Import filmu
   - Backend zapisuje oryginał w katalogu projektu.
   - Worker odczytuje metadane przez FFprobe.
   - System wylicza checksum pliku i zapisuje pełne parametry źródła.

2. Dekodowanie
   - Worker przetwarza film sekwencyjnie lub w shardach czasowych.
   - Analiza działa na pełnej rozdzielczości albo na skalowanej kopii z mapowaniem współrzędnych do oryginału.
   - Dla każdego sharda przechowywane są graniczne klatki, żeby tracking nie gubił osób na podziale.

3. Detekcja twarzy
   - YuNet znajduje twarze i landmarki bazowe.
   - Detekcje są rozszerzane marginesem bezpieczeństwa.
   - Niskie confidence nie powoduje odrzucenia regionu; słaba detekcja może nadal wygenerować blur albo flagę.

4. Rozpoznawanie allowlisty
   - SFace generuje embeddingi dla twarzy z wystarczającą jakością.
   - Embedding jest porównywany z bazą allowlisty przez cosine similarity.
   - Decyzja automatycznego odblurrowania wymaga przekroczenia wysokiego progu i braku konfliktu z inną osobą.
   - Twarz poniżej progu, z wieloma podobnymi kandydatami albo z niską jakością cropa pozostaje blurred lub uncertain.

5. Detekcja osoby
   - YOLO11 detect wykrywa sylwetki osób w klatkach.
   - YOLO11-seg generuje maski sylwetek dla trudniejszych ujęć lub przyszłych efektów na całej osobie.
   - Detekcje osób są trackowane niezależnie od twarzy przez tracker Ultralytics.
   - Face track jest kojarzony z person trackiem przez położenie twarzy wewnątrz boxa osoby, IoU i ciągłość czasową.

6. Estymacja głowy przy braku twarzy
   - YOLO11-pose wyznacza keypointy ciała, głowy i ramion, gdy są dostępne.
   - Jeśli keypointy głowy są widoczne, region redakcji powstaje z nosa, oczu, uszu i szyi z konserwatywnym marginesem.
   - Jeśli keypointy głowy nie są widoczne, region powstaje z górnej części person boxa oraz historii poprzednich klatek.
   - Każdy fallback bez widocznej twarzy generuje `ReviewFlag`.

7. Tracking i wygładzanie
   - ByteTrack łączy detekcje twarzy i osób w tracki.
   - Krótkie braki detekcji są interpolowane.
   - Regiony redakcji są wygładzane czasowo, żeby blur nie migotał.
   - Track jest zamykany przy długim braku detekcji albo ryzyku identity switch.

8. Reguły decyzji
   - Unknown face: blur twarzy.
   - Unknown person without face: blur przewidywanej głowy i flag.
   - Allowed face with high confidence: brak blura dla twarzy.
   - Allowed person temporarily without visible face: brak automatycznego blura tylko wtedy, gdy track jest ciągły, świeżo potwierdzony i bez konfliktu; fragment dostaje flagę do review.
   - Identity conflict: blur i flag.
   - Track crossing, crowd, occlusion, reflection, monitor screen, motion blur: blur i flag.

9. Generowanie masek
   - System tworzy finalną listę regionów redakcji na osi czasu.
   - Każdy region ma powód: unknown face, unknown head fallback, low confidence, identity conflict, manual override.
   - Maski są rozszerzane o margines bezpieczeństwa zależny od rozdzielczości i ruchu.

10. Review
   - Operator widzi listę flag posortowaną po ryzyku.
   - Operator może zmienić decyzję na tracku lub krótkim zakresie czasu.
   - Eksport jest dozwolony bez review tylko w trybie roboczym oznaczonym jako draft.
   - Eksport finalny wymaga zatwierdzenia wszystkich flag wysokiego ryzyka.

11. Render
   - FFmpeg renderuje finalny MP4 z oryginalnego źródła i zatwierdzonych regionów.
   - Pierwszy efekt to mocny Gaussian blur z featheringiem na krawędziach.
   - Audio jest kopiowane bez zmian w pierwszej wersji.
   - Eksport zapisuje manifest z wersjami modeli, decyzjami review i parametrami renderu.

## Enrollment allowlisty

Enrollment osoby z allowlisty wymaga zestawu zdjęć lub klatek z materiałów źródłowych. Minimalny zestaw startowy to 30 poprawnych cropów twarzy na osobę, rekomendowany zestaw produkcyjny to 100-300 cropów.

Zestaw powinien obejmować:

- przód, półprofil i profil;
- różne oświetlenie;
- okulary, czapkę, zarost, makijaż, mimikę;
- ujęcia z kamer i obiektywów podobnych do docelowych filmów;
- cropy z ruchu i lekkim rozmyciem.

System odrzuca próbki z wieloma twarzami, bardzo niską rozdzielczością, silnym zasłonięciem albo konfliktem z istniejącą allowlistą. Każde dodanie osoby wymaga zapisu podstawy zgody i identyfikacji operatora wykonującego enrollment.

## UI review

Ekrany pierwszej wersji:

- Lista projektów z postępem analizy i eksportów.
- Upload filmu z walidacją formatu i rozmiaru.
- Allowlista osób z enrollmentem, jakością próbek i historią zgód.
- Widok projektu z odtwarzaczem, overlayem regionów i timeline.
- Panel flag z filtrami: high risk, uncertain, identity conflict, fallback head, manual edits.
- Edycja tracka: blur, allow, uncertain, zakres czasowy, powód decyzji.
- Eksport z wyborem jakości, podglądem statusu i linkiem do lokalnego pliku wynikowego.

W odtwarzaczu operator powinien móc szybko przechodzić między flagami, porównywać klatkę przed i po redakcji oraz zobaczyć, dlaczego system podjął decyzję. UI nie powinno ukrywać niepewności modelu pod prostą etykietą typu "AI done".

## Efekty redakcji

Pierwszy efekt:

- mocny blur twarzy lub przewidywanej głowy;
- feathering krawędzi;
- minimalny margines bezpieczeństwa zależny od rozmiaru regionu;
- stabilizacja regionu między klatkami.

Architektura efektów:

- `blur_strong` jako efekt domyślny;
- `pixelate_strong` jako technicznie prosty wariant awaryjny;
- `solid_mask` jako tryb maksymalnej prywatności;
- `sticker_overlay` jako przyszły wariant dla twarzy Mikołaja, zajączka lub innych nakładek.

Efekt nie decyduje, co zasłonić. Decyzję podejmuje pipeline redakcji, a efekt tylko renderuje zatwierdzony region.

## Przygotowanie pod redakcję głosu

Pierwsza wersja zachowuje audio bez zmian, ale model danych i pipeline eksportu rozdzielają wideo od ścieżek audio. Manifest eksportu ma miejsce na przyszłe `AudioRedactionRegion`.

Przyszła redakcja głosu powinna być osobnym modułem:

- detekcja segmentów mowy;
- diarization speakerów;
- allowlista głosów;
- efekt zmiany głosu lub wyciszenia;
- osobny review timeline dla audio.

Nie implementujemy rozpoznawania głosu w pierwszym etapie, ponieważ zwiększa zakres prawny i testowy, a głównym celem jest redakcja wizualna.

## Walidacja jakości

Projekt wymaga własnego zestawu testowego z realnych albo bardzo podobnych materiałów. Zestaw walidacyjny powinien zawierać:

- dobre światło i słabe światło;
- tłum;
- ujęcia bokiem i tyłem;
- okulary, czapki, kaptury;
- ruch kamery;
- motion blur;
- częściowe zasłonięcia twarzy;
- osoby w tle;
- odbicia w lustrach lub szybach, jeśli występują w materiale;
- szybkie przejścia montażowe.

Metryki akceptacji:

- zero niezasłoniętych twarzy osób spoza allowlisty w zatwierdzonym eksporcie testowym;
- zero niezasłoniętych przewidywanych głów osób spoza allowlisty w ujęciach bez widocznej twarzy, jeśli osoba jest identyfikowalna;
- wszystkie identity conflict i fallback head pojawiają się na liście review;
- maski nie migoczą w typowych ujęciach;
- eksport zachowuje synchronizację audio-wideo;
- finalny render nie zmienia rozdzielczości ani FPS bez jawnej decyzji operatora.

## Fazy implementacji

### Faza 1: Fundament aplikacji

Zakres:

- Projekt Django z aplikacjami `projects`, `allowlist`, `analysis`, `review`, `registry` i SQLite jako bazą startową.
- Venv, `requirements.txt`, instrukcja `migrate` + `runserver` + `run_worker` w README.
- Upload filmu przez formularz Django, odczyt metadanych FFprobe, zapis projektu.
- Kolejka jobów (Django-Q2 lub Huey) i widoczny postęp analizy w widoku projektu.
- Model registry z ręcznie zarejestrowanymi modelami i hashami.
- Eksport testowy przez FFmpeg bez redakcji, żeby potwierdzić codec, kontener i synchronizację.

Kryterium zakończenia:

- Operator może wrzucić film, zobaczyć projekt, uruchomić job i dostać eksport roboczy zgodny technicznie ze źródłem.

### Faza 2: Twarze i allowlista

Zakres:

- Pobranie i rejestracja YuNet oraz SFace.
- Enrollment osoby z allowlisty.
- Detekcja twarzy na filmie.
- Embeddingi i dopasowanie do allowlisty.
- Reguła redakcji unknown face.
- Render mocnego blura dla twarzy spoza allowlisty.

Kryterium zakończenia:

- Na testowym filmie system blurrowuje wszystkie twarze spoza allowlisty i zostawia widoczną twarz youtubera w prostych ujęciach frontalnych.

### Faza 3: Fallback osoby i głowy

Zakres:

- Detekcja osób przez YOLO11 detect.
- Segmentacja sylwetek przez YOLO11-seg dla ujęć wymagających maski osoby.
- Estymacja pozy przez YOLO11-pose.
- Kojarzenie face tracków z person trackami.
- Wyznaczanie przewidywanego obszaru głowy dla osób bez widocznej twarzy.
- Flagi review dla fallbacku, konfliktów i słabych detekcji.

Kryterium zakończenia:

- Osoby tyłem, bokiem lub z zasłoniętą twarzą dostają blur przewidywanej głowy i pojawiają się w review.

### Faza 4: Review UI

Zakres:

- Odtwarzacz z overlayem regionów.
- Timeline tracków i flag.
- Zmiana decyzji na tracku lub zakresie czasu.
- Blokada eksportu finalnego przy niezatwierdzonych flagach wysokiego ryzyka.
- Audyt decyzji operatora.

Kryterium zakończenia:

- Operator może przejść po wszystkich flagach, poprawić decyzje i wyeksportować zatwierdzony film.

### Faza 5: Jakość renderu i stabilizacja

Zakres:

- Wygładzanie regionów w czasie.
- Interpolacja krótkich braków detekcji.
- Marginesy zależne od ruchu i rozdzielczości.
- Presety eksportu.
- Raport po eksporcie z listą modeli, flag i ręcznych decyzji.

Kryterium zakończenia:

- Blur jest stabilny, estetyczny i nie odsłania twarzy przy szybkich ruchach lub krótkich dropoutach detekcji.

### Faza 6: Hardening prostej produkcji z dostępem zewnętrznym

Zakres:

- Ten etap nie blokuje MVP; wracamy do niego dopiero po działającym przepływie upload, analiza, review i eksport.
- Uwierzytelnianie operatorów, sesje z rotacją, ochrona CSRF dla formularzy i rate limiting dla logowania.
- Limity uploadu, timeouty, walidacja MIME przez probing pliku i kwarantanna plików z błędnym kontenerem.
- Retencja danych źródłowych i eksportów.
- Logi audytowe.
- Obsługa błędów FFmpeg i GPU.
- Wznawianie przerwanych jobów.
- Testy regresji na zestawie walidacyjnym.
- Instrukcja aktualizacji modeli i odtwarzalności środowiska.

Kryterium zakończenia:

- Narzędzie nadaje się do regularnej pracy raz w tygodniu bez ręcznego grzebania w plikach lub procesach.

## Bezpieczeństwo i prywatność

Ten obszar jest hardeningiem po działającym MVP. Minimalnie nie serwujemy publicznie surowych plików przez statyczny katalog i utrzymujemy lokalny storage poza kodem repozytorium.

- Dostęp do aplikacji wymaga logowania.
- Aplikacja jest wystawiana bezpośrednio przez przekierowanie portu, bez reverse proxy i bez wymuszonego HTTPS.
- Panel administracyjny i endpointy operacyjne są dostępne tylko dla ról administracyjnych.
- Sesje operatorów mają krótką bezczynność, rotację tokenów i możliwość ręcznego unieważnienia.
- Logowanie ma rate limiting i blokadę po serii błędnych prób.
- Uploady mają limit rozmiaru, limit czasu, kontrolę rozszerzenia, kontrolę MIME i potwierdzenie przez FFprobe przed dopuszczeniem do analizy.
- Publiczny endpoint nie serwuje oryginałów ani artefaktów analizy bez autoryzacji.
- Hasła, tokeny sesji, filmy i dane formularzy będą przesyłane plaintextem, jeśli użytkownik łączy się przez HTTP przez internet; akceptujemy to jako świadomy kompromis prostoty wdrożenia.
- Oryginały i embeddingi są przechowywane lokalnie.
- Embeddingi twarzy traktujemy jako dane biometryczne.
- Każda osoba z allowlisty musi mieć zapisaną podstawę zgody.
- Eksport finalny zapisuje audyt decyzji redakcyjnych.
- Oryginały mają konfigurowalną retencję.
- Usunięcie projektu usuwa oryginał, artefakty analizy, maski, embeddingi projektu i eksporty.
- Logi nie zapisują pełnych klatek wideo, embeddingów, tokenów sesji, ścieżek signed URL ani danych formularzy uploadu.

## Główne ryzyka techniczne

- Detektor twarzy może gubić małe, boczne, zasłonięte lub rozmyte twarze.
- Face recognition może pomylić osobę przy niskiej jakości cropa albo podobnych twarzach.
- Tracking może zamienić identyfikatory osób przy przecięciach w tłumie.
- Fallback głowy może być niedokładny przy nietypowej pozie lub częściowej widoczności sylwetki.
- Estetyczny blur może być mniej prywatny niż solid mask przy bardzo małych regionach albo silnej kompresji.
- PyTorch, CUDA i sterowniki muszą być dopasowane do GPU Blackwell; bez warstwy kontenerów odpowiedzialność za zgodność wersji jest po stronie operatora hosta.

Odpowiedzi projektowe:

- Konserwatywne progi odblurrowania.
- Blur jako domyślna decyzja dla unknown i uncertain.
- Review flagi dla wszystkich trudnych przypadków.
- Marginesy bezpieczeństwa wokół twarzy i przewidywanej głowy.
- Model registry i walidacja na stałym zestawie testowym przed aktualizacją modeli.

## Kolejność pierwszych prac

1. Spisać dokładną politykę redakcji: kiedy allowed person bez widocznej twarzy zostaje widoczna, a kiedy trafia do blura.
2. Opublikować repozytorium jako AGPL-3.0 przed dodaniem zależności Ultralytics do wydania używanego produkcyjnie.
3. Zebrać 10-20 minut reprezentatywnego materiału testowego i przygotować ręcznie oczekiwane decyzje redakcyjne.
4. Przygotować lokalne środowisko: Python 3.12 w venv, sterowniki NVIDIA i CUDA pod Blackwell, PyTorch z odpowiednim wheelem CUDA, FFmpeg w PATH.
5. Zarejestrować modele YuNet, SFace, YOLO11 detect, YOLO11-seg i YOLO11-pose wraz z licencjami oraz hashami.
6. Zaimplementować upload, job queue i eksport techniczny.
7. Dodać detekcję twarzy, allowlistę i blur unknown faces.
8. Dodać fallback osób i głów.
9. Dodać review UI jako obowiązkową bramkę dla eksportu finalnego.
10. Przetestować pipeline na zestawie walidacyjnym i dostroić progi.
