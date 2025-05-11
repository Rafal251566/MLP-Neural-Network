using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;

namespace SiecNeuronowaMLP
{
    public class Neuron
    {
        public List<double> Wagi { get; set; }
        public double Bias { get; set; }
        public double Wyjscie { get; set; }
        public double PochodnaAktywacji { get; set; }

        public Neuron(int liczbaWejsc, bool useBias)
        {
            Wagi = new List<double>();
            Random random = new Random();
            for (int i = 0; i < liczbaWejsc; i++)
            {
                Wagi.Add(random.NextDouble() * 2 - 1); //wartosc z zakresu -1 : 1
            }
            if (useBias)
            {
                Bias = random.NextDouble() * 2 - 1;
            }
            else
            {
                Bias = 0;
            }
            Wyjscie = 0;
            PochodnaAktywacji = 0; // Inicjalizacja PochodnaAktywacji
        }

        public static double FunkcjaAktywacji(double x)
        {
            return 1 / (1 + Math.Exp(-x)); //beta = 1  unipolarna  
        }

        public static double PochodnaFunkcjiAktywacji(double x)
        {
            double sigmoid = FunkcjaAktywacji(x);
            return sigmoid * (1 - sigmoid);
        }

        public double ObliczWyjscie(List<double> wejscia)
        {
            double suma = 0;
            for (int i = 0; i < wejscia.Count; i++)
            {
                suma += wejscia[i] * Wagi[i];
            }

            if (Bias != 0)
            {
                suma += Bias;
            }

            Wyjscie = FunkcjaAktywacji(suma);
            PochodnaAktywacji = PochodnaFunkcjiAktywacji(suma);
            return Wyjscie;
        }
    }

    public class Warstwa
    {
        public List<Neuron> Neurony { get; set; }

        public Warstwa(int liczbaNeuronow, int wejsciaPoprzedniej, bool useBias)
        {
            Neurony = new List<Neuron>();
            for (int i = 0; i < liczbaNeuronow; i++)
            {
                Neurony.Add(new Neuron(wejsciaPoprzedniej, useBias));
            }
        }

        public List<double> ObliczWyjscia(List<double> wejscia)
        {
            return Neurony.Select(n => n.ObliczWyjscie(wejscia)).ToList();
        }
    }

    public class SiecNeuronowa
    {
        public List<Warstwa> Warstwy { get; set; }
        public bool UzywajBiasu { get; private set; }
        public double LearningRate { get; set; }
        public double Momentum { get; set; }
        public List<double> wszystkieBledy { get; set; }

        public SiecNeuronowa(int[] architektura, bool useBias, double learningRate = 0.1, double momentum = 0.9)
        {
            if (architektura == null || architektura.Length < 2)
            {
                throw new ArgumentException("Siec musi zawierać co najmniej warstwę wejściową i wyjściową.");
            }

            Warstwy = new List<Warstwa>();
            UzywajBiasu = useBias;
            LearningRate = learningRate;
            Momentum = momentum;
            wszystkieBledy = new List<double>();

            for (int i = 1; i < architektura.Length; i++)
            {
                int liczbaNeuronow = architektura[i];
                int liczbaWejscZPoprzedniejWarstwy = architektura[i - 1];
                Warstwy.Add(new Warstwa(liczbaNeuronow, liczbaWejscZPoprzedniejWarstwy, UzywajBiasu));
            }
        }

        public List<double> Propaguj(List<double> wejscia)
        {
            List<double> wyjscia = wejscia;
            foreach (var warstwa in Warstwy)
            {
                wyjscia = warstwa.ObliczWyjscia(wyjscia);
            }
            return wyjscia;
        }

        public void Trenuj(List<List<double>> daneWejsciowe, List<List<double>> oczekiwaneWyjscia, int liczbaEpok)
        {
            for (int epoka = 0; epoka < liczbaEpok; epoka++)
            {
                for (int m = 0; m < daneWejsciowe.Count; m++)
                {
                    // najpierw propagacja w przód
                    List<double> aktualneWyjscia = Propaguj(daneWejsciowe[m]);

                    // liczymy se blad
                    List<double> bledyWyjsciowe = ObliczBledy(oczekiwaneWyjscia[m], aktualneWyjscia);

                    // tera propagacja w tył
                    PropagujWstecz(bledyWyjsciowe, daneWejsciowe[m]);
                }

                // wyswietlanie bledu co 50 epok
                if (epoka % 50 == 0)
                {
                    double bladSredniokwadratowy = ObliczBladSredniokwadratowy(daneWejsciowe, oczekiwaneWyjscia);
                    wszystkieBledy.Add(bladSredniokwadratowy);
                    //Console.WriteLine($"Epoka: {epoka}, Błąd: {bladSredniokwadratowy}");

                    if(bladSredniokwadratowy < 0.016)
                    {
                        return;
                    }
                }

            }
        }

        private List<double> ObliczBledy(List<double> oczekiwaneWyjscia, List<double> aktualneWyjscia)
        {
            return oczekiwaneWyjscia.Zip(aktualneWyjscia, (oczekiwane, aktualne) => oczekiwane - aktualne).ToList();
        }

        private void PropagujWstecz(List<double> bledyWyjsciowe, List<double> wejscia)
        {
            List<double> bledyWarstwyNastępnej = bledyWyjsciowe;

            for (int i = Warstwy.Count - 1; i >= 0; i--)
            {
                List<double> bledyWarstwyBieżącej = new List<double>();
                List<double> wejsciaDoWarstwy = i > 0 ? Warstwy[i - 1].Neurony.Select(n => n.Wyjscie).ToList() : wejscia;

                for (int j = 0; j < Warstwy[i].Neurony.Count; j++)
                {
                    double delta;
                    if (i == Warstwy.Count - 1) // dla warsty wyjsciowej
                    {
                        delta = bledyWarstwyNastępnej[j] * Warstwy[i].Neurony[j].PochodnaAktywacji;
                    }
                    else // to dla warstwu ukrytych
                    {
                        double sumaDeltaWazona = 0;
                        for (int k = 0; k < Warstwy[i + 1].Neurony.Count; k++)
                        {
                            sumaDeltaWazona += Warstwy[i + 1].Neurony[k].Wagi[j] * bledyWarstwyNastępnej[k];
                        }
                        delta = sumaDeltaWazona * Warstwy[i].Neurony[j].PochodnaAktywacji;
                    }

                    bledyWarstwyBieżącej.Add(delta);

                    // aktualizuje wage
                    for (int k = 0; k < Warstwy[i].Neurony[j].Wagi.Count; k++)
                    {
                        Warstwy[i].Neurony[j].Wagi[k] += LearningRate * delta * wejsciaDoWarstwy[k] + Momentum * delta;
                    }

                    //a ti aktualizuje bias
                    if (UzywajBiasu)
                    {
                        Warstwy[i].Neurony[j].Bias += LearningRate * delta + Momentum * delta;
                    }
                }
                bledyWarstwyNastępnej = bledyWarstwyBieżącej;
            }
        }


        private double ObliczBladSredniokwadratowy(List<List<double>> daneWejsciowe, List<List<double>> oczekiwaneWyjscia)
        {
            double sumaBledow = 0;
            for (int m = 0; m < daneWejsciowe.Count; m++)
            {
                List<double> aktualneWyjscia = Propaguj(daneWejsciowe[m]);
                sumaBledow += aktualneWyjscia.Zip(oczekiwaneWyjscia[m], (aktualne, oczekiwane) => Math.Pow(aktualne - oczekiwane, 2)).Sum();
            }
            return sumaBledow / (daneWejsciowe.Count * oczekiwaneWyjscia[0].Count);
        }

        public void ZapiszSiec(string filePath)
        {
            using (StreamWriter writer = new StreamWriter(filePath))
            {
                writer.WriteLine(Warstwy.Count);
                writer.WriteLine(UzywajBiasu);
                writer.WriteLine(LearningRate);
                writer.WriteLine(Momentum);

                foreach (var warstwa in Warstwy)
                {
                    writer.WriteLine(warstwa.Neurony.Count);
                    foreach (var neuron in warstwa.Neurony)
                    {
                        writer.WriteLine(string.Join(";", neuron.Wagi));
                        writer.WriteLine(neuron.Bias);
                    }
                }
            }
        }

        public static SiecNeuronowa WczytajSiec(string filePath)
        {
            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException($"Plik '{filePath}' nie istnieje.");
            }

            using (StreamReader reader = new StreamReader(filePath))
            {
                int liczbaWarstw = int.Parse(reader.ReadLine());
                bool useBias = bool.Parse(reader.ReadLine());
                double learningRate = double.Parse(reader.ReadLine());
                double momentum = double.Parse(reader.ReadLine());
                int[] architektura = new int[liczbaWarstw + 1];

                List<List<List<double>>> wagiWarstw = new List<List<List<double>>>();
                List<List<double>> biasyWarstw = new List<List<double>>();

                for (int i = 0; i < liczbaWarstw; i++)
                {
                    int liczbaNeuronow = int.Parse(reader.ReadLine());
                    architektura[i + 1] = liczbaNeuronow;
                    List<List<double>> wagiNeuronow = new List<List<double>>();
                    List<double> biasyNeuronow = new List<double>();
                    for (int j = 0; j < liczbaNeuronow; j++)
                    {
                        wagiNeuronow.Add(reader.ReadLine().Split(';').Select(double.Parse).ToList());
                        biasyNeuronow.Add(double.Parse(reader.ReadLine()));
                    }
                    wagiWarstw.Add(wagiNeuronow);
                    biasyWarstw.Add(biasyNeuronow);
                }

                if (liczbaWarstw > 0 && wagiWarstw.Any() && biasyWarstw.Any())
                {
                    int rozmiarWejscia = wagiWarstw[0].First().Count;
                    architektura[0] = rozmiarWejscia;

                    SiecNeuronowa siec = new SiecNeuronowa(architektura, useBias, learningRate, momentum);

                    for (int i = 0; i < liczbaWarstw; i++)
                    {
                        for (int j = 0; j < architektura[i + 1]; j++)
                        {
                            if (wagiWarstw.Count > i && wagiWarstw[i].Count > j)
                            {
                                siec.Warstwy[i].Neurony[j].Wagi = wagiWarstw[i][j];
                            }
                            if (biasyWarstw.Count > i && biasyWarstw[i].Count > j)
                            {
                                siec.Warstwy[i].Neurony[j].Bias = biasyWarstw[i][j];
                            }
                        }
                    }
                    return siec;
                }
            }
            return null;
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            string filePath = "dane.txt";
            List<List<double>> daneWejscioweNauka = new List<List<double>>();
            List<List<double>> oczekiwaneWyjsciaNauka = new List<List<double>>();
            List<List<double>> daneWejscioweTest = new List<List<double>>();
            List<List<double>> oczekiwaneWyjsciaTest = new List<List<double>>();
            Dictionary<string, List<double>> etykietyKlucze = new Dictionary<string, List<double>>();
            int indeksEtykiety = 0;
            Dictionary<string, List<List<double>>> danePosortowane = new Dictionary<string, List<List<double>>>();

            try
            {
                List<string> linie = File.ReadAllLines(filePath).ToList();

                foreach (var linia in linie)
                {
                    string[] pola = linia.Split(',');
                    if (pola.Length == 5)
                    {
                        List<double> cechy = pola.Take(4)
                            .Select(s => double.Parse(s.Replace(',', '.'), CultureInfo.InvariantCulture))
                            .ToList();
                        string etykieta = pola[4];

                        if (!etykietyKlucze.ContainsKey(etykieta))
                        {
                            etykietyKlucze[etykieta] = new List<double> { 0, 0, 0 };
                            etykietyKlucze[etykieta][indeksEtykiety++] = 1;
                            danePosortowane[etykieta] = new List<List<double>>();
                        }
                        danePosortowane[etykieta].Add(cechy);
                    }
                }

                foreach (var gatunekData in danePosortowane)
                {
                    string gatunek = gatunekData.Key;
                    List<List<double>> daneGatunku = gatunekData.Value;
                    List<double> kluczEtykiety = etykietyKlucze[gatunek];

                    for (int i = 0; i < daneGatunku.Count; i++)
                    {
                        if (i < 40)
                        {
                            daneWejscioweNauka.Add(daneGatunku[i]);
                            oczekiwaneWyjsciaNauka.Add(kluczEtykiety.ToList());
                        }
                        else
                        {
                            daneWejscioweTest.Add(daneGatunku[i]);
                            oczekiwaneWyjsciaTest.Add(kluczEtykiety.ToList());
                        }
                    }
                }

                Random random = new Random();
                List<int> indeksy = Enumerable.Range(0, daneWejscioweNauka.Count).ToList();
                indeksy = indeksy.OrderBy(x => random.Next()).ToList();

                List<List<double>> poszaflowaneDaneWejscioweNauka = new List<List<double>>();
                List<List<double>> poszaflowaneOczekiwaneWyjsciaNauka = new List<List<double>>();

                foreach (int indeks in indeksy)
                {
                    poszaflowaneDaneWejscioweNauka.Add(daneWejscioweNauka[indeks]);
                    poszaflowaneOczekiwaneWyjsciaNauka.Add(oczekiwaneWyjsciaNauka[indeks]);
                }

                Random random2 = new Random();
                List<int> indekxy = Enumerable.Range(0, daneWejscioweTest.Count).ToList();
                indeksy = indekxy.OrderBy(x => random2.Next()).ToList();

                List<List<double>> poszaflowaneDaneWejscioweTest = new List<List<double>>();
                List<List<double>> poszaflowaneOczekiwaneWyjsciaTest = new List<List<double>>();

                foreach (int indeks in indeksy)
                {
                    poszaflowaneDaneWejscioweTest.Add(daneWejscioweTest[indeks]);
                    poszaflowaneOczekiwaneWyjsciaTest.Add(oczekiwaneWyjsciaTest[indeks]);
                }

                daneWejscioweNauka = poszaflowaneDaneWejscioweNauka;
                oczekiwaneWyjsciaNauka = poszaflowaneOczekiwaneWyjsciaNauka;

                daneWejscioweTest = poszaflowaneDaneWejscioweTest;
                oczekiwaneWyjsciaTest = poszaflowaneOczekiwaneWyjsciaTest;

                int[] architektura = { 4, 5, 3 }; // 4 wejścia 5 neuronów ukrytych 3 wyjścia
                bool useBias = true;
                double learningRate = 0.4;
                double momentum = 0.6;
                int liczbaEpok = 10001;

                SiecNeuronowa siec = new SiecNeuronowa(architektura, useBias, learningRate, momentum);


                siec.Trenuj(daneWejscioweNauka, oczekiwaneWyjsciaNauka, liczbaEpok);

                Console.WriteLine("\nWyniki testowania:");
                List<int> przewidywaneEtykiety = new List<int>();
                List<int> rzeczywisteEtykiety = new List<int>();

                for (int i = 0; i < daneWejscioweTest.Count; i++)
                {
                    List<double> wyjscia = siec.Propaguj(daneWejscioweTest[i]);
                    int przewidywanyIndeks = wyjscia.IndexOf(wyjscia.Max());
                    int oczekiwanyIndeks = oczekiwaneWyjsciaTest[i].IndexOf(oczekiwaneWyjsciaTest[i].Max());

                    string przewidywanaKlasa = zmienNaIndex(przewidywanyIndeks);
                    string oczekiwanaKlasa = zmienNaIndex(oczekiwanyIndeks);

                    Console.WriteLine($"Wejście: [{string.Join(", ", daneWejscioweTest[i])}] -> Przewidywana: {przewidywanaKlasa}, Oczekiwana: {oczekiwanaKlasa}");

                    przewidywaneEtykiety.Add(przewidywanyIndeks);
                    rzeczywisteEtykiety.Add(oczekiwanyIndeks);
                }

                
                int poprawneOdpowiedzi = przewidywaneEtykiety.Zip(rzeczywisteEtykiety, (przewidywany, rzeczywisty) => przewidywany == rzeczywisty).Count(x => x);
                Console.WriteLine($"\nLiczba poprawnie sklasyfikowanych obiektów: {poprawneOdpowiedzi}/{daneWejscioweTest.Count}");

                
                int liczbaKlas = etykietyKlucze.Count;
                var confusionMatrix = new int[liczbaKlas, liczbaKlas];

                for (int i = 0; i < przewidywaneEtykiety.Count; i++)
                {
                    int przewidywany = przewidywaneEtykiety[i];
                    int rzeczywisty = rzeczywisteEtykiety[i];
                    confusionMatrix[rzeczywisty, przewidywany]++;
                }

                Console.WriteLine("\nMacierz pomyłek:");
                for (int i = 0; i < liczbaKlas; i++)
                {
                    for (int j = 0; j < liczbaKlas; j++)
                    {
                        Console.Write(confusionMatrix[i, j] + "\t");
                    }
                    Console.WriteLine();
                }

                for (int i = 0; i < liczbaKlas; i++)
                {
                    int truePositive = confusionMatrix[i, i];
                    int falsePositive = 0;
                    int falseNegative = 0;

                    for (int j = 0; j < liczbaKlas; j++)
                    {
                        if (j != i)
                        {
                            falsePositive += confusionMatrix[j, i];
                            falseNegative += confusionMatrix[i, j];
                        }
                    }

                    double precision = (truePositive + falsePositive == 0) ? 0 : (double)truePositive / (truePositive + falsePositive);
                    double recall = (truePositive + falseNegative == 0) ? 0 : (double)truePositive / (truePositive + falseNegative);
                    double fMeasure = (precision + recall == 0) ? 0 : 2 * (precision * recall) / (precision + recall);

                    Console.WriteLine($"\nKlasa {zmienNaIndex(i)}:");
                    Console.WriteLine($"Precision: {precision:F2}");
                    Console.WriteLine($"Recall: {recall:F2}");
                    Console.WriteLine($"F-measure: {fMeasure:F2}");
                }

                double accuracy = (double)poprawneOdpowiedzi / daneWejscioweTest.Count;
                Console.WriteLine($"\nDokładność testu: {accuracy:F2}");

                string saveBledy = "wszystkieBledy.txt";
                File.WriteAllLines(saveBledy, siec.wszystkieBledy.Select(b => b.ToString(CultureInfo.InvariantCulture)));

                string savePath = "siec_iris.txt";
                siec.ZapiszSiec(savePath);
                SiecNeuronowa wczytanaSiec = SiecNeuronowa.WczytajSiec(savePath);

                string relativePath = Path.Combine("..", "..", "..", "..", "Neural MLP Network", "bin", "Debug", "net9.0-windows", "Neural MLP Network.exe");
                string fullPath = Path.GetFullPath(relativePath);

                if (File.Exists(fullPath))
                {
                    Process.Start(fullPath);
                }
                else
                {
                    Console.WriteLine("Nie znaleziono pliku View.exe! Sprawdź ścieżkę.");
                }
            }
            catch (FileNotFoundException)
            {
                Console.WriteLine($"Plik '{filePath}' nie został znaleziony.");
            }
            catch (IOException ex)
            {
                Console.WriteLine($"Wystąpił błąd podczas odczytu pliku: {ex.Message}");
            }
            catch (FormatException)
            {
                Console.WriteLine("Błąd formatowania danych w pliku.");
            }


            List<List<double>> daneWejscioweAutoenkoder = new List<List<double>>()
            {
                new List<double> { 1, 0, 0, 0 },
                new List<double> { 0, 1, 0, 0 },
                new List<double> { 0, 0, 1, 0 },
                new List<double> { 0, 0, 0, 1 }
            };
            List<List<double>> oczekiwaneWyjsciaAutoenkoder = new List<List<double>>()
            {
                new List<double> { 1, 0, 0, 0 },
                new List<double> { 0, 1, 0, 0 },
                new List<double> { 0, 0, 1, 0 },
                new List<double> { 0, 0, 0, 1 }
            };

            int liczbaEpokAutoenkoder = 1000;
            double learningRateAutoenkoder = 0.6;
            double momentumAutoenkoder = 0.0;

            Console.WriteLine("\n--- Trening Autoenkodera z Biasem ---");
            int[] architekturaZBiasem = { 4, 2, 4 };
            SiecNeuronowa siecZBiasem = new SiecNeuronowa(architekturaZBiasem, true, learningRateAutoenkoder, momentumAutoenkoder);
            siecZBiasem.Trenuj(daneWejscioweAutoenkoder, oczekiwaneWyjsciaAutoenkoder, liczbaEpokAutoenkoder);

            Console.WriteLine("\nWyjścia sieci z biasem po treningu:");
            for (int i = 0; i < daneWejscioweAutoenkoder.Count; i++)
            {
                List<double> wyjscia = siecZBiasem.Propaguj(daneWejscioweAutoenkoder[i]);
                Console.WriteLine($"Wejście: [{string.Join(", ", daneWejscioweAutoenkoder[i])}] -> Wyjście: [{string.Join(", ", wyjscia.Select(x => x.ToString("F3")))}]");
            }

            // Trening bez biasu w warstwach ukrytych i wyjściowej
            Console.WriteLine("\n--- Trening Autoenkodera BEZ Biasu ---");
            SiecNeuronowa siecBezBiasu = new SiecNeuronowa(architekturaZBiasem, false, learningRateAutoenkoder, momentumAutoenkoder);
            siecBezBiasu.Trenuj(daneWejscioweAutoenkoder, oczekiwaneWyjsciaAutoenkoder, liczbaEpokAutoenkoder);

            Console.WriteLine("\nWyjścia sieci bez biasu po treningu:");
            for (int i = 0; i < daneWejscioweAutoenkoder.Count; i++)
            {
                List<double> wyjscia = siecBezBiasu.Propaguj(daneWejscioweAutoenkoder[i]);
                Console.WriteLine($"Wejście: [{string.Join(", ", daneWejscioweAutoenkoder[i])}] -> Wyjście: [{string.Join(", ", wyjscia.Select(x => x.ToString("F3")))}]");
            }

            List<Tuple<double, double>> konfiguracje = new List<Tuple<double, double>>()
            {
                Tuple.Create(0.9, 0.0),
                Tuple.Create(0.6, 0.0),
                Tuple.Create(0.2, 0.0),
                Tuple.Create(0.9, 0.6),
                Tuple.Create(0.2, 0.9)
            };

            foreach (var konfiguracja in konfiguracje)
            {
                double learningRate = konfiguracja.Item1;
                double momentum = konfiguracja.Item2;

                Console.WriteLine($"\n--- Trening Autoenkodera z Biasem (LR={learningRate}, Momentum={momentum}) ---");
                SiecNeuronowa siec = new SiecNeuronowa(architekturaZBiasem, true, learningRate, momentum);
                siec.Trenuj(daneWejscioweAutoenkoder, oczekiwaneWyjsciaAutoenkoder, liczbaEpokAutoenkoder);

                Console.WriteLine("\nWyjścia sieci po treningu:");
                for (int i = 0; i < daneWejscioweAutoenkoder.Count; i++)
                {
                    List<double> wyjscia = siec.Propaguj(daneWejscioweAutoenkoder[i]);
                    Console.WriteLine($"Wejście: [{string.Join(", ", daneWejscioweAutoenkoder[i])}] -> Wyjście: [{string.Join(", ", wyjscia.Select(x => x.ToString("F3")))}]");
                }

                // Opcjonalnie: Zapisywanie błędów dla każdej konfiguracji
                string bledyFilePath = $"bledy_lr{learningRate}_momentum{momentum}.txt";
                File.WriteAllLines(bledyFilePath, siec.wszystkieBledy.Select(b => b.ToString(CultureInfo.InvariantCulture)));
                Console.WriteLine($"Błędy zapisano do pliku: {bledyFilePath}");
            }
        }

        static string zmienNaIndex(int indeks)
        {
            switch (indeks)
            {
                case 0: return "0";
                case 1: return "1";
                case 2: return "2";
                default: return "Nieznana";
            }
        }
    }
}