using MLP_Neural_Network;
using System.Diagnostics;
using System.Globalization;

namespace SiecNeuronowaMLP
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Wybierz opcję:");
            Console.WriteLine("1 - Uczenie i testowanie sieci dla danych Iris");
            Console.WriteLine("2 - Uczenie autoenkodera");
            Console.WriteLine("3 - Wczytanie sieci z pliku");
            Console.Write("Twój wybór: ");

            string? wybor = Console.ReadLine();

            switch (wybor)
            {
                case "1":
                    UruchomIrysy();
                    break;
                case "2":
                    UruchomAutoenkoder();
                    break;
                case "3":
                    DataLoader loader = new();
                    Console.WriteLine("Wpisz nazwę pliku: ");
                    string path = Console.ReadLine();
                    var siec = SiecNeuronowa.WczytajSiec(path);
                    loader.ReadDataFromFile();
                    TestujSiec(siec, loader);
                    break;
                default:
                    Console.WriteLine("Nieprawidłowy wybór.");
                    break;
            }

            Console.WriteLine("\nKoniec programu.");
        }

        static void UruchomIrysy()
        {
            DataLoader loader = new();
            try
            {
                loader.ReadDataFromFile();
                ShuffleData(ref loader.daneWejscioweTest, ref loader.oczekiwaneWyjsciaTest);

                Console.WriteLine("\nKonfiguracja sieci dla danych Iris:");

                Console.Write("Podaj architekturę sieci (liczby neuronów w warstwach oddzielone przecinkami, np. 4,5,3): ");
                string? architekturaStr = Console.ReadLine();
                int[] architekturaMLP = architekturaStr?.Split(',').Select(int.Parse).ToArray() ?? new int[] { 4, 5, 3 };

                Console.Write("Podaj learning rate (np. 0.4): ");
                double lrMLP = double.TryParse(Console.ReadLine(), NumberStyles.Float, CultureInfo.InvariantCulture, out var lr) ? lr : 0.4;

                Console.Write("Czy używać momentum? (tak/nie): ");
                bool uzywajMomentum = Console.ReadLine()?.ToLower() == "tak";
                double momentumMLP = 0.0;
                if (uzywajMomentum)
                {
                    Console.Write("Podaj wartość momentum (np. 0.6): ");
                    momentumMLP = double.TryParse(Console.ReadLine(), NumberStyles.Float, CultureInfo.InvariantCulture, out var mom) ? mom : 0.6;
                }

                Console.Write("Podaj liczbę epok (np. 10001): ");
                int epokiMLP = int.TryParse(Console.ReadLine(), out var epoki) ? epoki : 10001;

                Console.Write("Podaj wartość błędu sieci przy którym zatrzyma się nauka (np. 0.015): ");
                double bladSieci = 0.0;
                bladSieci = double.TryParse(Console.ReadLine(), NumberStyles.Float, CultureInfo.InvariantCulture, out var blad) ? blad : 0.015;

                Console.Write("Czy używać biasu? (tak/nie): ");
                bool uzywajBiasu = Console.ReadLine()?.ToLower() == "tak";

                Console.Write("Czy nauka ma być prowadzona przy losowej kolejnosci podawania wzorców? (tak/nie) ");
                bool losowaKolejnosc = Console.ReadLine()?.ToLower() == "tak";

                var siec = new SiecNeuronowa(architekturaMLP, uzywajBiasu, lrMLP, momentumMLP);
                Console.WriteLine("\nRozpoczęto trening sieci dla danych Iris...");
                siec.Trenuj(loader.daneWejscioweNauka, loader.oczekiwaneWyjsciaNauka, epokiMLP,losowaKolejnosc, bladSieci);
                Console.WriteLine("Trening zakończony.");
                TestujSiec(siec, loader);

                File.WriteAllLines("wszystkieBledy_iris.txt", siec.wszystkieBledy.Select(b => b.ToString(CultureInfo.InvariantCulture)));
                siec.ZapiszSiec("siec_iris.txt");

                if (File.Exists("../../../../Neural MLP Network/bin/Debug/net9.0-windows/Neural MLP Network.exe"))
                {
                        Process.Start(Path.GetFullPath("../../../../Neural MLP Network/bin/Debug/net9.0-windows/Neural MLP Network.exe"),"wszystkieBledy_iris.txt Irysów");
                }

            }
            catch (FileNotFoundException) { Console.WriteLine($"Plik '{DataLoader.filePath}' nie został znaleziony."); }
            catch (IOException ex) { Console.WriteLine($"Błąd I/O: {ex.Message}"); }
            catch (FormatException) { Console.WriteLine("Błąd formatowania danych."); }
            catch (Exception ex) { Console.WriteLine($"Wystąpił nieoczekiwany błąd: {ex.Message}"); }
        }

        static void UruchomAutoenkoder()
        {
            int[] architekturaAutoenkoder = { 4, 2, 4 };
            double lrAuto = 0.6, momentumAuto = 0.0;
            int epokiAuto = 1000;
            var daneAuto = new List<List<double>> {
                new() { 1, 0, 0, 0 }, new() { 0, 1, 0, 0 },
                new() { 0, 0, 1, 0 }, new() { 0, 0, 0, 1 }
            };
            var oczekiwaneAuto = new List<List<double>>(daneAuto);

            Console.WriteLine("\n\n--- Autoenkoder z biasem ---");
            TrenujAutoenkoder(architekturaAutoenkoder, true, lrAuto, momentumAuto, epokiAuto, daneAuto, oczekiwaneAuto);

            Console.WriteLine("\n--- Autoenkoder bez biasu ---");
            TrenujAutoenkoder(architekturaAutoenkoder, false, lrAuto, momentumAuto, epokiAuto, daneAuto, oczekiwaneAuto);

            var konfiguracje = new List<(double lr, double momentum)> {
                (0.9, 0.0), (0.6, 0.0), (0.2, 0.0), (0.9, 0.6), (0.2, 0.9)
            };

            foreach (var (lr, mom) in konfiguracje)
            {
                string sciezka = $"bledy_lr{lr}_momentum{mom}_auto.txt";
                string info = $"Autoenkoder_(LR={lr}_Momentum={mom})";
                Console.WriteLine($"\n\n--- Autoenkoder (LR={lr}, Momentum={mom}) ---");
                TrenujAutoenkoder(architekturaAutoenkoder, true, lr, mom, epokiAuto, daneAuto, oczekiwaneAuto, sciezka, info);
            }
        }

        private static void TrenujAutoenkoder(
            int[] architektura,
            bool useBias,
            double lr,
            double momentum,
            int epoki,
            List<List<double>> dane,
            List<List<double>> oczekiwane,
            string zapisSciezka = null,
            string info = null
            )
        {
            SiecNeuronowa siec = new(architektura, useBias, lr, momentum);
            Console.WriteLine("\nRozpoczęto trening autoenkodera...");
            siec.Trenuj(dane, oczekiwane, epoki,true, 0.016);
            Console.WriteLine("Trening autoenkodera zakończony.");

            Console.WriteLine("\nWyjścia sieci po treningu:");
            for (int i = 0; i < dane.Count; i++)
            {
                var wyjscia = siec.Propaguj(dane[i]);
                Console.WriteLine($"\nWejście: [{string.Join(", ", dane[i])}] -> Wyjście: [{string.Join(", ", wyjscia.Select(x => x.ToString("F3")))}]");

                Console.WriteLine(" \n Wyjścia warstwy ukrytej:");
                var wyjsciaWarstwyUkrytej = siec.Warstwy[0].Neurony.Select(n => n.Wyjscie).ToList();
                Console.WriteLine($"  [{string.Join(", ", wyjsciaWarstwyUkrytej.Select(x => x.ToString("F3")))}]");
            }


            if (zapisSciezka != null)
            {
                File.WriteAllLines(zapisSciezka, siec.wszystkieBledy.Select(b => b.ToString(CultureInfo.InvariantCulture)));
                Console.WriteLine($"Błędy zapisano do pliku: {zapisSciezka}");
            }

            if (File.Exists("../../../../Neural MLP Network/bin/Debug/net9.0-windows/Neural MLP Network.exe") && zapisSciezka != null)
            {
                Process.Start(Path.GetFullPath("../../../../Neural MLP Network/bin/Debug/net9.0-windows/Neural MLP Network.exe"), $"{zapisSciezka} {info}");
            }
        }

        private static void ObliczMacierzPomyłek(List<int> przewidywane, List<int> rzeczywiste, DataLoader loader)
        {
            int liczbaKlas = loader.etykietyKlucze.Count;
            var matrix = new int[liczbaKlas, liczbaKlas];

            for (int i = 0; i < przewidywane.Count; i++)
                matrix[rzeczywiste[i], przewidywane[i]]++;

            Console.WriteLine("\nMacierz pomyłek:");
            for (int i = 0; i < liczbaKlas; i++)
            {
                for (int j = 0; j < liczbaKlas; j++)
                    Console.Write(matrix[i, j] + "\t");
                Console.WriteLine();
            }

            int poprawne = przewidywane.Zip(rzeczywiste, (p, r) => p == r).Count(x => x);
            double accuracy = (double)poprawne / przewidywane.Count;
            Console.WriteLine($"\nDokładność testu: {accuracy:F2}");

            for (int i = 0; i < liczbaKlas; i++)
            {
                int tp = matrix[i, i];
                int fp = Enumerable.Range(0, liczbaKlas).Where(j => j != i).Sum(j => matrix[j, i]);
                int fn = Enumerable.Range(0, liczbaKlas).Where(j => j != i).Sum(j => matrix[i, j]);

                double precision = tp + fp == 0 ? 0 : (double)tp / (tp + fp);
                double recall = tp + fn == 0 ? 0 : (double)tp / (tp + fn);
                double f = precision + recall == 0 ? 0 : 2 * precision * recall / (precision + recall);

                Console.WriteLine($"\nKlasa {loader.zmienNaIndex(i)}:");
                Console.WriteLine($"Precision: {precision:F2}");
                Console.WriteLine($"Recall: {recall:F2}");
                Console.WriteLine($"F-measure: {f:F2}");
            }
        }

        private static void TestujSiec(SiecNeuronowa siec, DataLoader loader)
        {
            Console.WriteLine("\nWyniki testowania z wyznaczeniem błędu dla każdego wzorca:");
            List<int> przewidywane = new();
            List<int> rzeczywiste = new();

            string sciezkaDoPliku = "wyniki_testu.txt";

            // Wyczyść plik przed rozpoczęciem testowania
            File.WriteAllText(sciezkaDoPliku, string.Empty);

            using (StreamWriter sw = new StreamWriter(sciezkaDoPliku, true)) // Teraz dopisujemy do wyczyszczonego pliku
            {
                for (int i = 0; i < loader.daneWejscioweTest.Count; i++)
                {
                    var wyjscia = siec.Propaguj(loader.daneWejscioweTest[i]);
                    var oczekiwane = loader.oczekiwaneWyjsciaTest[i];
                    int przew = wyjscia.IndexOf(wyjscia.Max());
                    int oczek = oczekiwane.IndexOf(oczekiwane.Max());

                    double blad = wyjscia.Zip(oczekiwane, (o, e) => Math.Pow(o - e, 2)).Sum();
                    sw.WriteLine("------------------------------------------------------------------------------------------------------------------------------------------------------------------------");
                    sw.WriteLine($"Wejście: [{string.Join(", ", loader.daneWejscioweTest[i])}]");
                    sw.WriteLine($"\nSuma kwadratów błędów: {blad:F4}");
                    sw.WriteLine($"\nOczekiwane wyjście: [{string.Join(", ", oczekiwane)}]");
                    var bledyKonkretne = wyjscia.Zip(oczekiwane, (o, e) => Math.Pow(o - e, 2)).ToList();
                    sw.WriteLine($"\nBłędy konkretne: [{string.Join(", ", bledyKonkretne.Select(b => b.ToString("F6")))}]");
                    sw.WriteLine($"\nWyjście sieci: [{string.Join(", ", wyjscia.Select(x => x.ToString("F3")))}]");
                    sw.WriteLine("\nWagi warstwy wyjsciowej:");
                    var wagiWarstwyWysciowej = siec.Warstwy[1].Neurony.Select(n => n.Wagi.Select(w => w.ToString("F3")).ToList()).ToList();
                    sw.WriteLine($"[{string.Join(", ", wagiWarstwyWysciowej.SelectMany(x => x))}]");
                    sw.WriteLine("\nWyjścia warstwy ukrytej:");
                    var wyjsciaWarstwyUkrytej = siec.Warstwy[0].Neurony.Select(n => n.Wyjscie).ToList();
                    sw.WriteLine($"[{string.Join(", ", wyjsciaWarstwyUkrytej.Select(x => x.ToString("F3")))}]");

                    sw.WriteLine("\nWagi warstwy ukrytej:");
                    for (int j = siec.Warstwy.Count - 1; j > 0; j--)
                    {
                        var wagiWarstwyUkrytej = siec.Warstwy[j].Neurony.Select(n => n.Wagi.Select(w => w.ToString("F3")).ToList()).ToList();
                        sw.WriteLine($"[{string.Join(", ", wagiWarstwyUkrytej.SelectMany(x => x))}]");
                    }
                    sw.WriteLine("------------------------------------------------------------------------------------------------------------------------------------------------------------------------");
                    sw.WriteLine(); // Dodaj pustą linię po każdym wzorcu
                                    //Console.WriteLine($"\nWzorzec {i + 1}:");
                                    //Console.WriteLine($"  Przewidywana klasa: {loader.zmienNaIndex(przew)}");
                                    //Console.WriteLine($"  Oczekiwana klasa: {loader.zmienNaIndex(oczek)}");

                    przewidywane.Add(przew);
                    rzeczywiste.Add(oczek);
                }
            }

            Console.WriteLine($"Dane zostały zapisane do pliku: {sciezkaDoPliku}");
            ObliczMacierzPomyłek(przewidywane, rzeczywiste, loader);
        }

        public static void ShuffleData(ref List<List<double>> dane, ref List<List<double>> etykiety)
        {
            Random random = new Random();
            List<int> indeksy = Enumerable.Range(0, dane.Count).OrderBy(_ => random.Next()).ToList();

            var noweDane = new List<List<double>>();
            var noweEtykiety = new List<List<double>>();

            foreach (int i in indeksy)
            {
                noweDane.Add(dane[i]);
                noweEtykiety.Add(etykiety[i]);
            }

            dane = noweDane;
            etykiety = noweEtykiety;
        }
    }
}