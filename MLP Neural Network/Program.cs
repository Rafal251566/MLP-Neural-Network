using System;
using System.Collections.Generic;
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

                // wyswietlanie bledu co 100 epok
                if (epoka % 100 == 0)
                {
                    double bladSredniokwadratowy = ObliczBladSredniokwadratowy(daneWejsciowe, oczekiwaneWyjscia);
                    Console.WriteLine($"Epoka: {epoka}, Błąd: {bladSredniokwadratowy}");
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

            //NA TEn moment test na danych wpisanych z palucha :D
            List<List<double>> daneWejscioweNauka = new List<List<double>>
            {
                new List<double> { 5.1, 3.5, 1.4, 0.2 }, // 0
                new List<double> { 7.0, 3.2, 4.7, 1.4 }, // 1
                new List<double> { 4.9, 3.0, 1.4, 0.2 }, // 0
                new List<double> { 4.7, 3.2, 1.3, 0.2 }, // 0
                new List<double> { 5.7, 2.8, 4.5, 1.3 }, // 1

                new List<double> { 4.6, 3.1, 1.5, 0.2 }, // 0
                new List<double> { 5.0, 3.6, 1.4, 0.2 }, // 0
                new List<double> { 4.6, 3.4, 1.4, 0.3 }, // 0
                new List<double> { 6.4, 3.2, 4.5, 1.5 }, // 1
                new List<double> { 5.0, 3.4, 1.5, 0.2 }, // 0

                new List<double> { 4.4, 2.9, 1.4, 0.2 }, // 0
                new List<double> { 6.3, 3.3, 6.0, 2.5 }, // 2
                new List<double> { 6.9, 3.1, 4.9, 1.5 }, // 1
                new List<double> { 5.5, 2.3, 4.0, 1.3 }, // 1
                new List<double> { 6.5, 2.8, 4.6, 1.5 }, // 1

                new List<double> { 4.9, 3.1, 1.5, 0.1 }, // 0
                new List<double> { 6.3, 3.3, 4.7, 1.6 }, // 1
                new List<double> { 4.9, 2.4, 3.3, 1.0 }, // 1
                new List<double> { 5.8, 2.7, 5.1, 1.9 }, // 2
                new List<double> { 7.1, 3.0, 5.9, 2.1 }, // 2

                new List<double> { 5.2, 2.7, 3.9, 1.4 }, // 1
                new List<double> { 6.3, 2.9, 5.6, 1.8 }, // 2
                new List<double> { 6.6, 2.9, 4.6, 1.3 }, // 1
                new List<double> { 6.5, 3.0, 5.8, 2.2 }, // 2
                new List<double> { 7.6, 3.0, 6.6, 2.1 }, // 2

                new List<double> { 5.4, 3.9, 1.7, 0.4 }, // 0
                new List<double> { 4.9, 2.5, 4.5, 1.7 }, // 2
                new List<double> { 7.3, 2.9, 6.3, 1.8 }, // 2
                new List<double> { 6.7, 2.5, 5.8, 1.8 }, // 2
                new List<double> { 7.2, 3.6, 6.1, 2.5 }  // 2
            };

            // Kodowanie 1-z-K dla wyjsc bo 3 klasy irys
            List<List<double>> oczekiwaneWyjsciaNauka = new List<List<double>>
            {
                new List<double> { 1, 0, 0 },
                new List<double> { 0, 1, 0 },
                new List<double> { 1, 0, 0 },
                new List<double> { 1, 0, 0 },
                new List<double> { 0, 1, 0 },

                new List<double> { 1, 0, 0 },
                new List<double> { 1, 0, 0 },
                new List<double> { 1, 0, 0 },
                new List<double> { 0, 1, 0 },
                new List<double> { 1, 0, 0 },

                new List<double> { 1, 0, 0 },
                new List<double> { 0, 0, 1 },
                new List<double> { 0, 1, 0 },
                new List<double> { 0, 1, 0 },
                new List<double> { 0, 1, 0 },

                new List<double> { 1, 0, 0 },
                new List<double> { 0, 1, 0 },
                new List<double> { 0, 1, 0 },
                new List<double> { 0, 0, 1 },
                new List<double> { 0, 0, 1 },

                new List<double> { 0, 1, 0 },
                new List<double> { 0, 0, 1 },
                new List<double> { 0, 1, 0 },
                new List<double> { 0, 0, 1 },
                new List<double> { 0, 0, 1 },

                new List<double> { 1, 0, 0 },
                new List<double> { 0, 0, 1 },
                new List<double> { 0, 0, 1 },
                new List<double> { 0, 0, 1 },
                new List<double> { 0, 0, 1 } 
            };

            List<List<double>> daneWejsciowe = new List<List<double>>
            {
                new List<double> { 5.5, 2.6, 4.4, 1.2 }, // 1
                new List<double> { 5.0, 3.5, 1.3, 0.3 }, // 0
                new List<double> { 4.5, 2.3, 1.3, 0.3 }, // 0
                new List<double> { 4.4, 3.2, 1.3, 0.2 }, // 0
                new List<double> { 5.0, 3.5, 1.6, 0.6 }, // 0

                new List<double> { 6.7, 3.0, 5.2, 2.3 }, // 2
                new List<double> { 4.8, 3.0, 1.4, 0.3 }, // 0
                new List<double> { 5.1, 3.8, 1.6, 0.2 }, // 0
                new List<double> { 4.6, 3.2, 1.4, 0.2 }, // 0
                new List<double> { 5.3, 3.7, 1.5, 0.2 }, // 0

                new List<double> { 5.0, 3.3, 1.4, 0.2 }, // 0
                new List<double> { 5.8, 2.7, 5.1, 1.9 }, // 2
                new List<double> { 6.1, 3.0, 4.6, 1.4 }, // 1
                new List<double> { 5.8, 2.6, 4.0, 1.2 }, // 1
                new List<double> { 5.0, 2.3, 3.3, 1.0 }, // 1

                new List<double> { 5.1, 3.8, 1.9, 0.4 }, // 0
                new List<double> { 5.6, 2.7, 4.2, 1.3 }, // 1
                new List<double> { 5.7, 3.0, 4.2, 1.2 }, // 1
                new List<double> { 5.7, 2.9, 4.2, 1.3 }, // 1
                new List<double> { 6.2, 2.9, 4.3, 1.3 }, // 1

                new List<double> { 5.1, 2.5, 3.0, 1.1 }, // 1
                new List<double> { 6.7, 3.1, 5.6, 2.4 }, // 2
                new List<double> { 6.9, 3.1, 5.1, 2.3 }, // 2
                new List<double> { 6.8, 3.2, 5.9, 2.3 }, // 2
                new List<double> { 6.7, 3.3, 5.7, 2.5 }, // 2

                new List<double> { 5.7, 2.8, 4.1, 1.3 }, // 1
                new List<double> { 6.3, 2.5, 5.0, 1.9 }, // 2
                new List<double> { 6.5, 3.0, 5.2, 2.0 }, // 2
                new List<double> { 6.2, 3.4, 5.4, 2.3 }, // 2
                new List<double> { 5.9, 3.0, 5.1, 1.8 }  // 2
            };

            List<List<double>> oczekiwaneWyjscia = new List<List<double>>
            {
                new List<double> { 0, 1, 0 },
                new List<double> { 1, 0, 0 }, 
                new List<double> { 1, 0, 0 }, 
                new List<double> { 1, 0, 0 }, 
                new List<double> { 1, 0, 0 },

                new List<double> { 0, 0, 1 },
                new List<double> { 1, 0, 0 }, 
                new List<double> { 1, 0, 0 }, 
                new List<double> { 1, 0, 0 }, 
                new List<double> { 1, 0, 0 }, 

                new List<double> { 1, 0, 0 },
                new List<double> { 0, 0, 1 },
                new List<double> { 0, 1, 0 }, 
                new List<double> { 0, 1, 0 }, 
                new List<double> { 0, 1, 0 },

                new List<double> { 1, 0, 0 },
                new List<double> { 0, 1, 0 }, 
                new List<double> { 0, 1, 0 }, 
                new List<double> { 0, 1, 0 }, 
                new List<double> { 0, 1, 0 }, 

                new List<double> { 0, 1, 0 }, 
                new List<double> { 0, 0, 1 }, 
                new List<double> { 0, 0, 1 }, 
                new List<double> { 0, 0, 1 }, 
                new List<double> { 0, 0, 1 },

                new List<double> { 0, 1, 0 },
                new List<double> { 0, 0, 1 }, 
                new List<double> { 0, 0, 1 }, 
                new List<double> { 0, 0, 1 }, 
                new List<double> { 0, 0, 1 }  
            };



            int[] architektura = { 4, 10, 3 }; // 4 wejścia 10 neuronów ukrytych 3 wyjścia
            bool useBias = true;
            double learningRate = 0.1;
            double momentum = 0.9;
            int liczbaEpok = 1001;

            SiecNeuronowa siec = new SiecNeuronowa(architektura, useBias, learningRate, momentum);

            siec.Trenuj(daneWejscioweNauka, oczekiwaneWyjsciaNauka, liczbaEpok);

            Console.WriteLine("Wyniki testowania:");
            for (int i = 0; i < daneWejsciowe.Count; i++)
            {
                List<double> wyjscia = siec.Propaguj(daneWejsciowe[i]);
                int przewidywanyIndeks = wyjscia.IndexOf(wyjscia.Max());

                int oczekiwanyIndeks = oczekiwaneWyjscia[i].IndexOf(oczekiwaneWyjscia[i].Max());

                string przewidywanaKlasa = zmienNaIndex(przewidywanyIndeks);
                string oczekiwanaKlasa = zmienNaIndex(oczekiwanyIndeks);

                Console.WriteLine($"Wejście: [{string.Join(", ", daneWejsciowe[i])}] -> Przewidywana: {przewidywanaKlasa}, Oczekiwana: {oczekiwanaKlasa}");
            }

            string savePath = "siec_iris.txt";
            siec.ZapiszSiec(savePath);
            SiecNeuronowa wczytanaSiec = SiecNeuronowa.WczytajSiec(savePath);
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