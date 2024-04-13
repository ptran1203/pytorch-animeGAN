import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from losses import LossSummary
from utils import load_checkpoint, save_checkpoint


class Trainer:
    """
    Base Trainer class
    """

    def __init__(
        self,
        generator,
        discriminator,
        lr_G: float = 0.001,
        lr_D: float = 0.001,
        num_workers: int = None,
        device = "cuda:0",
    ) -> None:
        self.G = generator
        self.D = discriminator
        self.num_workers = num_workers
        self.optimizer_g = optim.Adam(self.G.parameters(), lr=lr_G, betas=(0.5, 0.999))
        self.optimizer_d = optim.Adam(self.G.parameters(), lr=lr_D, betas=(0.5, 0.999))
        self.loss_tracker = LossSummary()
        self.device = torch.device(device)

    def init_weight_G(self, weight: str):
        """Init Generator weight"""
        return load_checkpoint(self.G, weight)

    def init_weight_D(self, weight: str):
        """Init Discriminator weight"""
        return load_checkpoint(self.D, weight)

    def pretrain_generator(self, train_dataset):
        pass

    def train_epoch(self, epoch, train_loader):
        for data in 
    def train(
        self,
        train_dataset: Dataset,
        init_generator: str = None,
        init_discriminator: str = None,
        pretrain_generator: bool = False,
        epochs: int = 100,
        batch_size: int = 8,
    ):
        """
        Train Generator and Discriminator.
        """
        # Set up dataloader
        data_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            collate_fn=collate_fn,
        )
        start_epoch = self.init_weight_G()

        if pretrain_generator:
            self.pretrain_generator(train_dataset)


        for epoch in range(start_epoch, epochs):
            self.train_epoch(epoch, train_dataset)



    G = Generator(args.dataset).cuda()
    D = Discriminator(args).cuda()

    loss_tracker = LossSummary()

    loss_fn = AnimeGanLoss(args)

    # Create DataLoader
    data_loader = DataLoader(
        AnimeDataSet(args),
        batch_size=args.batch_size,
        num_workers=cpu_count(),
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer_g = optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    start_e = 0
    if args.resume == 'GD':
        # Load G and D
        try:
            start_e = load_checkpoint(G, args.checkpoint_dir)
            print("G weight loaded")
            load_checkpoint(D, args.checkpoint_dir)
            print("D weight loaded")
        except Exception as e:
            print('Could not load checkpoint, train from scratch', e)
    elif args.resume == 'G':
        # Load G only
        try:
            start_e = load_checkpoint(G, args.checkpoint_dir, posfix='_init')
        except Exception as e:
            print('Could not load G init checkpoint, train from scratch', e)

    for e in range(start_e, args.epochs):
        print(f"Epoch {e}/{args.epochs}")
        bar = tqdm(data_loader)
        G.train()

        init_losses = []

        if e < args.init_epochs:
            # Train with content loss only
            set_lr(optimizer_g, args.init_lr)
            for img, *_ in bar:
                img = img.cuda()
                
                optimizer_g.zero_grad()

                fake_img = G(img)
                loss = loss_fn.content_loss_vgg(img, fake_img)
                loss.backward()
                optimizer_g.step()

                init_losses.append(loss.cpu().detach().numpy())
                avg_content_loss = sum(init_losses) / len(init_losses)
                bar.set_description(f'[Init Training G] content loss: {avg_content_loss:2f}')

            set_lr(optimizer_g, args.lr_g)
            save_checkpoint(G, optimizer_g, e, args, posfix='_init')
            save_samples(G, data_loader, args, subname='initg')
            continue

        loss_tracker.reset()
        for img, anime, anime_gray, anime_smt_gray in bar:
            # To cuda
            img = img.cuda()
            anime = anime.cuda()
            anime_gray = anime_gray.cuda()
            anime_smt_gray = anime_smt_gray.cuda()

            # ---------------- TRAIN D ---------------- #
            optimizer_d.zero_grad()
            fake_img = G(img).detach()

            # Add some Gaussian noise to images before feeding to D
            if args.d_noise:
                fake_img += gaussian_noise()
                anime += gaussian_noise()
                anime_gray += gaussian_noise()
                anime_smt_gray += gaussian_noise()

            fake_d = D(fake_img)
            real_anime_d = D(anime)
            real_anime_gray_d = D(anime_gray)
            real_anime_smt_gray_d = D(anime_smt_gray)

            loss_d = loss_fn.compute_loss_D(
                fake_d, real_anime_d, real_anime_gray_d, real_anime_smt_gray_d)

            loss_d.backward()
            optimizer_d.step()

            loss_tracker.update_loss_D(loss_d)

            # ---------------- TRAIN G ---------------- #
            optimizer_g.zero_grad()

            fake_img = G(img)
            fake_d = D(fake_img)

            adv_loss, con_loss, gra_loss, col_loss = loss_fn.compute_loss_G(
                fake_img, img, fake_d, anime_gray)

            loss_g = adv_loss + con_loss + gra_loss + col_loss

            loss_g.backward()
            optimizer_g.step()

            loss_tracker.update_loss_G(adv_loss, gra_loss, col_loss, con_loss)

            avg_adv, avg_gram, avg_color, avg_content = loss_tracker.avg_loss_G()
            avg_adv_d = loss_tracker.avg_loss_D()
            bar.set_description(f'loss G: adv {avg_adv:2f} con {avg_content:2f} gram {avg_gram:2f} color {avg_color:2f} / loss D: {avg_adv_d:2f}')

        if e % args.save_interval == 0:
            save_checkpoint(G, optimizer_g, e, args)
            save_checkpoint(D, optimizer_d, e, args)
            save_samples(G, data_loader, args)


if __name__ == '__main__':
    args = parse_args()

    print("# ==== Train Config ==== #")
    for arg in vars(args):
        print(arg, getattr(args, arg))
    print("==========================")

    main(args)
