namespace MLP_Neural_Network
{
    class SiecNeuronowa
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

                    if (bladSredniokwadratowy < 0.016)
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

                    // aktualizuje wage z uwzględnieniem momentum
                    for (int k = 0; k < Warstwy[i].Neurony[j].Wagi.Count; k++)
                    {
                        double zmianaWagi = LearningRate * delta * wejsciaDoWarstwy[k] + Momentum * Warstwy[i].Neurony[j].PoprzednieZmianyWag[k];
                        Warstwy[i].Neurony[j].Wagi[k] += zmianaWagi;
                        Warstwy[i].Neurony[j].PoprzednieZmianyWag[k] = zmianaWagi;
                    }

                    // aktualizuje bias z uwzględnieniem momentum
                    if (UzywajBiasu)
                    {
                        double zmianaBiasu = LearningRate * delta + Momentum * Warstwy[i].Neurony[j].PoprzedniaZmianaBiasu;
                        Warstwy[i].Neurony[j].Bias += zmianaBiasu;
                        Warstwy[i].Neurony[j].PoprzedniaZmianaBiasu = zmianaBiasu;
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
}
