using MLP_Neural_Network;
using System.Diagnostics;
using System.Globalization;

namespace SiecNeuronowaMLP
{
    class Program
    {
        static void Main(string[] args)
        {
            DataLoader loader = new();
            int[] architekturaMLP = { 4, 5, 3 };
            int[] architekturaAutoenkoder = { 4, 2, 4 };

            double lrMLP = 0.4, momentumMLP = 0.6;
            int epokiMLP = 10001;
            double lrAuto = 0.6, momentumAuto = 0.0;
            int epokiAuto = 1000;

            try
            {
                loader.ReadDataFromFile();

                ShuffleData(ref loader.daneWejscioweNauka, ref loader.oczekiwaneWyjsciaNauka);
                ShuffleData(ref loader.daneWejscioweTest, ref loader.oczekiwaneWyjsciaTest);

                var siec = new SiecNeuronowa(architekturaMLP, true, lrMLP, momentumMLP);
                siec.Trenuj(loader.daneWejscioweNauka, loader.oczekiwaneWyjsciaNauka, epokiMLP);
                TestujSiec(siec, loader);

                File.WriteAllLines("wszystkieBledy.txt", siec.wszystkieBledy.Select(b => b.ToString(CultureInfo.InvariantCulture)));
                siec.ZapiszSiec("siec_iris.txt");

                if (File.Exists("../../../../Neural MLP Network/bin/Debug/net9.0-windows/Neural MLP Network.exe"))
                    Process.Start(Path.GetFullPath("../../../../Neural MLP Network/bin/Debug/net9.0-windows/Neural MLP Network.exe"));

            }
            catch (FileNotFoundException) { Console.WriteLine($"Plik '{DataLoader.filePath}' nie został znaleziony."); }
            catch (IOException ex) { Console.WriteLine($"Błąd I/O: {ex.Message}"); }
            catch (FormatException) { Console.WriteLine("Błąd formatowania danych."); }

            var daneAuto = new List<List<double>> {
                new() { 1, 0, 0, 0 }, new() { 0, 1, 0, 0 },
                new() { 0, 0, 1, 0 }, new() { 0, 0, 0, 1 }
            };
            var oczekiwaneAuto = new List<List<double>>(daneAuto);

            Console.WriteLine("\n--- Autoenkoder z biasem ---");
            TrenujAutoenkoder(architekturaAutoenkoder, true, lrAuto, momentumAuto, epokiAuto, daneAuto, oczekiwaneAuto);

            Console.WriteLine("\n--- Autoenkoder bez biasu ---");
            TrenujAutoenkoder(architekturaAutoenkoder, false, lrAuto, momentumAuto, epokiAuto, daneAuto, oczekiwaneAuto);

            var konfiguracje = new List<(double lr, double momentum)> {
                (0.9, 0.0), (0.6, 0.0), (0.2, 0.0), (0.9, 0.6), (0.2, 0.9)
            };

            foreach (var (lr, mom) in konfiguracje)
            {
                string sciezka = $"bledy_lr{lr}_momentum{mom}.txt";
                Console.WriteLine($"\n--- Autoenkoder (LR={lr}, Momentum={mom}) ---");
                TrenujAutoenkoder(architekturaAutoenkoder, true, lr, mom, epokiAuto, daneAuto, oczekiwaneAuto, sciezka);
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
            string? zapisSciezka = null
            )
        {
            SiecNeuronowa siec = new(architektura, useBias, lr, momentum);
            siec.Trenuj(dane, oczekiwane, epoki);

            Console.WriteLine("\nWyjścia sieci po treningu:");
            for (int i = 0; i < dane.Count; i++)
            {
                var wyjscia = siec.Propaguj(dane[i]);
                Console.WriteLine($"Wejście: [{string.Join(", ", dane[i])}] -> Wyjście: [{string.Join(", ", wyjscia.Select(x => x.ToString("F3")))}]");
            }

            if (zapisSciezka != null)
            {
                File.WriteAllLines(zapisSciezka, siec.wszystkieBledy.Select(b => b.ToString(CultureInfo.InvariantCulture)));
                Console.WriteLine($"Błędy zapisano do pliku: {zapisSciezka}");
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
            Console.WriteLine("\nWyniki testowania:");
            List<int> przewidywane = new();
            List<int> rzeczywiste = new();

            for (int i = 0; i < loader.daneWejscioweTest.Count; i++)
            {
                var wyjscia = siec.Propaguj(loader.daneWejscioweTest[i]);
                int przew = wyjscia.IndexOf(wyjscia.Max());
                int oczek = loader.oczekiwaneWyjsciaTest[i].IndexOf(loader.oczekiwaneWyjsciaTest[i].Max());

                Console.WriteLine($"Wejście: [{string.Join(", ", loader.daneWejscioweTest[i])}] -> Przewidywana: {loader.zmienNaIndex(przew)}," +
                    $" Oczekiwana: {loader.zmienNaIndex(oczek)}");

                przewidywane.Add(przew);
                rzeczywiste.Add(oczek);
            }

            ObliczMacierzPomyłek(przewidywane, rzeczywiste, loader);
        }

        private static void ShuffleData(ref List<List<double>> dane, ref List<List<double>> etykiety)
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