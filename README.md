
Autoencoders
=================

=================
**Parametry**
=================

--dropout - prawdopodobieństwo wyłączenia neuronu w fazie uczenia (dla 1 próbki uczącej)
--max_iterations - maksymalna liczba iteracji na warstwę
--restart_gradient_after - liczba iteracji po których algorytm gradientów sprzężonych jest restartowany
--epsilon - RMS na zbiorze uczącym przy którym uczenie warstwy jest kończone
--learning_coefficient - standardowo rozumiany współczynnik uczenia (określa rozmiar kroku)
--regularization_factor - współczynnik skalowania wag
--batches_at_supervised - 
--configuration - liczby neuronów w warstwach ukrytych oddzielone przecinkami - np. 10,14,15 oznacza sieć o 3 warstwach ukrytych, mających odpowiednio 10, 14 i 15 neuronów
--print_at - liczba iteracji co ile aktualny RMS drukowany jest na standardowe wyjście
--generate - czy należy wygenerować zbiór uczący i testowy? Generowane zbiory są separowalne liniowo
--inputs - liczba wejść
--outputs - liczba wyjść
--test_data - ścieżka do pliku z danymi testowymi (w przypadku użycia --generate plik zostanie nadpisany)
--train_data - ścieżka do pliku z danymi testowymi (w przypadku użycia --generate plik zostanie nadpisany)
--train_instances - liczba instancji w zbiorze uczącym
--test_instances - liczba instancji w zbiorze testowym
--config_file - ścieżka do pliku konfiguracyjnego (plik konfiguracyjny zawiera parametry w postaci <nazwa>=<wartość> oddzielone znakami nowej linii)

=================
**Dane**
=================

Dane wejściowe muszą być w formacie CSV oddzielonym przecinkami, bez nagłówków.
Separatorem części ułamkowej musi być kropka.

=================
**Instalacja**
=================

Wspierane jest tylko budowanie ze źródeł - wymaga zainstalowanej lokalnie bilbioteki boost oraz kompilatora C++ zgodnego z C++11 co najmniej tak, jak kompilator wbudowany w MSVS2013.
Aplikacja była testowana wyłącznie na systemie Windows, ale w razie problemów w uruchomieniu na innych systemach, proszę o kontakt.

=================
**Budowanie**
=================
Wymagany jest CMake i boost (program_options, ublas i unit_test_framework).
wykonanie komend (z katalogu projektu):
    mkdir build
	cd build
	cmake ..
	#Visual
	msbuild /m ALL_BUILD.sln
	#GCC/Clang
	make -j3
	
powinno skutkować utworzeniem katalogu /bin/exe (lub /bin/exe/Release | /bin/exe/Debug)
mogą wystąpić problemy z kolejnością linkowania bibliotek boost - należy ją wtedy dostosować (odpowiednio do typu konfiguracji)