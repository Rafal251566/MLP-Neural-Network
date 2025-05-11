
namespace MLP_Neural_Network
{
    class Warstwa
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
}
