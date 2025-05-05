using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;


//WERSJA POCZĄTKOWA WIELE DO ZMIANY ZEBY ZADANIE MIALO SENS A I ROZDIELIC KLASY NA PLIKI CHYBA ŁADNIEJ
namespace SiecNeuronowaMLP
{
    public class Neuron
    {
        public List<double> Wagi { get; set; }
        public double Bias { get; set; }
        public double Wyjscie { get; set; }

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
        }

        public static double FunkcjaAktywacji(double x)
        {
            return 1 / (1 + Math.Exp(-x)); //beta = 1
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

        public SiecNeuronowa(int[] architektura, bool useBias)
        {
            if (architektura == null || architektura.Length < 2)
            {
                throw new ArgumentException("Siec musi zawierać co najmniej warstwę wejściową i wyjściową.");
            }

            Warstwy = new List<Warstwa>();
            UzywajBiasu = useBias;

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

        public void ZapiszSiec(string filePath)
        {
            using (StreamWriter writer = new StreamWriter(filePath))
            {
                writer.WriteLine(Warstwy.Count);
                writer.WriteLine(UzywajBiasu);

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
                int[] architektura = new int[liczbaWarstw + 1];

                List<List<List<double>>> wagiWarstw = new List<List<List<double>>>(); //lista warstw,lista wag dla neuronu, konkretna waga
                List<List<double>> biasyWarstw = new List<List<double>>(); //lista warstw, wartosc biasow

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

                    SiecNeuronowa siec = new SiecNeuronowa(architektura, useBias);

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

        class Program
        {
            static void Main(string[] args)
            {
                int[] architektura = { 2, 3, 1 }; // 2 wejścia pozniej 3 neurony w warstwie i  1 wyjście
                bool useBias = true;

                SiecNeuronowa siec = new SiecNeuronowa(architektura, useBias);

                List<double> wejscia = new List<double> { 0.5, 0.1 }; //testowe przykladowe wejscia
                List<double> wyjscia = siec.Propaguj(wejscia);
                Console.WriteLine($"Wyjście sieci dla wejść [{string.Join(", ", wejscia)}]: [{string.Join(", ", wyjscia)}]");

                string savePath = "siec.txt";
                siec.ZapiszSiec(savePath);
                Console.WriteLine($"Sieć zapisana do pliku: {savePath}");

                string looadPath = "siec.txt";
                SiecNeuronowa wczytanaSiec = SiecNeuronowa.WczytajSiec(looadPath);
                if (wczytanaSiec != null)
                {
                    Console.WriteLine($"Sieć wczytana z pliku: {looadPath}");
                    List<double> wyjsciaWczytanejSieci = wczytanaSiec.Propaguj(wejscia);
                    Console.WriteLine($"Wyjście wczytnaej sieci dla wejść [{string.Join(", ", wejscia)}]: [{string.Join(", ", wyjsciaWczytanejSieci)}]");
                }
                else
                {
                    Console.WriteLine("Nie udało się wczytać pliku.");
                }
            }
        }
    }
}